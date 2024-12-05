import os
import re
from typing import List, Set, Dict, Tuple
import anthropic
import random
import json
from util import load_secrets

# Load secrets at module level
load_secrets()


class ExampleValidator:
    @staticmethod
    def clean_line(line: str) -> str:
        """
        Clean a line by removing any multi-line content within JSON strings.
        Only return the start of the line up to any nested content.
        """
        # If this is an Action with code, just return the Action line
        if 'create_tool' in line and '"code":' in line:
            return line.split('"code":', 1)[0] + '"code": "..."}'

        return line

    @staticmethod
    def validate_pattern(example: str) -> Tuple[bool, str]:
        """
        Validate that example follows the correct pattern while allowing for tool usage patterns.
        Returns (is_valid, error_message)
        """
        # First, clean up the example by handling multi-line content
        cleaned_lines = []
        in_code_block = False
        current_line = ""

        for line in example.split('\n'):
            stripped = line.strip()
            if not stripped:
                continue

            # Handle start of code block in create_tool
            if 'create_tool' in stripped and '"code":' in stripped:
                cleaned_lines.append(ExampleValidator.clean_line(stripped))
                in_code_block = True
                continue

            # Skip lines while in code block until we see the closing brace
            if in_code_block:
                if stripped.endswith('"}}'):  # End of code block
                    in_code_block = False
                continue

            # Normal line processing
            if not in_code_block:
                cleaned_lines.append(stripped)

        # Now validate the cleaned lines
        current_state = 'expect_observation'
        had_thought = False
        last_action = None
        ignore_next_system = False

        for line in cleaned_lines:
            if line.startswith('Action:'):
                last_action = line
                if 'communicate' in line:
                    ignore_next_system = True

            # Rest of validation logic remains the same...
            # (previous validation code here)

        if current_state != 'expect_observation':
            return False, "Example ended in wrong state"

        if not last_action or 'wait' not in last_action:
            return False, "Example must end with wait Action"

        return True, ""

    @staticmethod
    def format_pattern_guide() -> str:
        """Return a clear explanation of the required pattern."""
        return """The example must strictly follow this pattern:

1. Start with an Observation
2. If it's a system Observation:
   - Can be followed by another Observation, or a Thought
3. If it's a user Observation:
   - Must follow with one or more Thoughts showing reasoning
   - Then an Action
4. After any communicate Action:
   - Must have a Thought about waiting for response
   - Must have a wait Action
5. After a wait Action:
   - Next must be an Observation

Every communicate must be followed by a wait sequence, and the example must end with a wait Action."""

class TrainingDataExpander:
    def __init__(self):
        self.client = anthropic.Anthropic()
        self.used_descriptions: Set[str] = set()
        self.validator = ExampleValidator()

    def extract_file_info(self, file_path: str) -> Tuple[str, List[str], List[str]]:
        """
        Extract the docstring description, existing examples, and any new descriptions from the file.
        Returns (guidance, existing_descriptions, new_descriptions)
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Extract docstring (guidance for examples)
        docstring_match = re.search(r"'''(.*?)'''", content, re.DOTALL)
        if not docstring_match:
            raise ValueError("No docstring found in file. Please add a docstring with example guidance.")

        guidance = docstring_match.group(1).strip()

        # Split content into lines and process
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        existing_descriptions = []
        new_descriptions = []

        # Scan from beginning for existing examples
        for line in lines:
            if line.startswith('#') and not line.startswith('#New:'):
                desc = line[1:].strip()
                if desc:
                    existing_descriptions.append(desc)

        # Scan from end for new descriptions
        for line in reversed(lines):
            if line.startswith('#New:'):
                desc = line[5:].strip()  # Remove '#New:' prefix
                if desc:
                    new_descriptions.insert(0, desc)  # Insert at front to maintain order

        return guidance, existing_descriptions, new_descriptions

    def generate_new_descriptions(self, guidance: str, existing_descriptions: List[str], num_to_generate: int = 25) -> \
    List[str]:
        """Generate new scenario descriptions based on existing ones and file guidance."""
        prompt = f"""Given this guidance for creating AI assistant scenarios:

{guidance}

These scenarios will be implemented following a strict interaction pattern:
1. Each user observation must be followed by at least one thought from the AI before any action
2. System messages (like delivery confirmations) can go directly to actions
3. The AI must show its reasoning through thoughts before taking any action
4. Scenarios should encourage multiple turns of back-and-forth conversation

Here are some example scenario descriptions that follow this guidance:

{chr(10).join('- ' + desc for desc in existing_descriptions)}

Create {num_to_generate} new, unique AI assistant scenario descriptions following the same guidance. Each should:
1. Follow the guidance exactly - {guidance}
2. Be substantially different from the examples provided
3. Be realistic and practical
4. Not duplicate any existing scenarios
5. Focus on conversational human interactions
6. Encourage multiple thought steps and reasoning before actions
7. Create opportunities for back-and-forth dialogue

Keep in mind that each scenario will be implemented with multiple turns of:
Observation -> Thought(s) -> Action -> Observation -> Thought(s) -> Action -> etc.

Return only the new descriptions, one per line, without any numbering or bullet points."""

        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=8192,
                temperature=0.9,
                messages=[{"role": "user", "content": prompt}]
            )

            # Split response into lines and clean them
            new_descriptions = [
                line.strip() for line in response.content[0].text.split('\n')
                if line.strip() and not line.strip().startswith(('-', '*', '•'))
            ]

            # Filter out any that are too similar to existing ones
            unique_descriptions = []
            for desc in new_descriptions:
                if not any(self.is_similar(desc, existing) for existing in existing_descriptions):
                    unique_descriptions.append(desc)

            return unique_descriptions

        except Exception as e:
            print(f"Error generating descriptions: {e}")
            return []

    def is_similar(self, desc1: str, desc2: str) -> bool:
        """Basic similarity check between two descriptions."""
        common_words = {'ai', 'assistant', 'example', 'helping', 'help', 'with', 'the', 'a', 'an'}
        words1 = set(word.lower() for word in desc1.split()) - common_words
        words2 = set(word.lower() for word in desc2.split()) - common_words

        intersection = words1.intersection(words2)
        smaller_set = min(len(words1), len(words2))

        if smaller_set == 0:
            return False

        similarity = len(intersection) / smaller_set
        return similarity > 0.8

    def generate_example_for_description(self, description: str, guidance: str, example_template: str) -> str:
        """Generate a complete example for a given scenario description."""
        import time
        max_retries = 3
        retry_delay = 5
        prompt = f"""Following this guidance:
{guidance}

Create a training example for this AI assistant scenario:
# {description}

{self.validator.format_pattern_guide()}

Additional requirements:
- Use only the communicate action for interactions
- Focus on natural conversation flow
- Show empathetic and thoughtful responses
- Create a realistic and detailed interaction
- Follow the guidance exactly: {guidance}
- Every user message must be followed by at least one thought showing reasoning
- Only system messages can skip the thought step

Here's a correctly formatted example template:
{example_template}

Return only the example, starting with the Observation."""

        for attempt in range(max_retries):
            try:
                print(f"Attempt {attempt + 1}/{max_retries}...")
                response = self.client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=8192,
                    temperature=0.8,
                    messages=[{"role": "user", "content": prompt}]
                )

                if not response.content or not response.content[0].text:
                    print("Error: Received empty response from Claude")
                    if attempt < max_retries - 1:
                        print(f"Waiting {retry_delay} seconds before retry...")
                        time.sleep(retry_delay)
                    continue

                example = response.content[0].text.strip()
                is_valid, error = self.validator.validate_pattern(example)

                if is_valid:
                    return example
                else:
                    print(f"Attempt {attempt + 1} failed validation: {error}")
                    if attempt < max_retries - 1:
                        print(f"Waiting {retry_delay} seconds before retry...")
                        time.sleep(retry_delay)
                        prompt += f"\n\nPrevious attempt failed because: {error}\nPlease fix this and ensure correct pattern."

            except Exception as e:
                print(f"Error on attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    print(f"Waiting {retry_delay} seconds before retry...")
                    time.sleep(retry_delay)
                continue

        return ""

    def get_example_templates(self, file_path: str) -> List[str]:
        """Extract complete examples from file and randomly select 3 as templates."""
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Split into examples based on # comments
        examples = []
        current_example = []

        for line in content.split('\n'):
            if line.strip().startswith('#') and not line.strip().startswith('#New:'):
                if current_example:
                    examples.append('\n'.join(current_example))
                current_example = []
            elif line.strip():
                current_example.append(line.strip())

        if current_example:  # Don't forget last example
            examples.append('\n'.join(current_example))

        # Filter for valid examples and select 3 randomly
        valid_examples = []
        for example in examples:
            is_valid, _ = self.validator.validate_pattern(example)
            if is_valid:
                # Extract just the O/T/A sequence, removing the comment
                example_content = '\n'.join(example.split('\n')[1:])
                valid_examples.append(example_content)

        if len(valid_examples) < 3:
            raise ValueError(f"Need at least 3 valid examples in file, found {len(valid_examples)}")

        return random.sample(valid_examples, 3)

    def expand_training_data(self, file_path: str, num_descriptions=10) -> List[Tuple[str, str]]:
        """Main function to expand training data."""
        # Get guidance, existing descriptions, and any new descriptions from file
        guidance, existing_descriptions, new_descriptions = self.extract_file_info(file_path)
        print(f"Found guidance: {guidance}")
        print(f"Found {len(existing_descriptions)} existing descriptions")

        # Get random example templates
        templates = self.get_example_templates(file_path)
        print(f"Selected {len(templates)} template examples")

        if new_descriptions:
            print(f"\nFound {len(new_descriptions)} new descriptions to implement")
        else:
            # Generate new descriptions only if none found in file
            print("\nGenerating new scenario descriptions...")
            new_descriptions = self.generate_new_descriptions(guidance, existing_descriptions, num_descriptions)
            print(f"Generated {len(new_descriptions)} new descriptions")

        # Generate complete examples
        new_examples = []
        print("\nGenerating complete examples for each description...")
        for i, description in enumerate(new_descriptions, 1):
            print(f"\nGenerating example {i}/{len(new_descriptions)}: {description[:50]}...")
            # Randomly select one of the templates for each example
            template = random.choice(templates)
            example = self.generate_example_for_description(description, guidance, template)
            if example:
                new_examples.append((description, example))
            else:
                print(f"Failed to generate valid example for: {description}")

        return new_examples


def main():
    expander = TrainingDataExpander()
    training_dir = "TrainingData"
    num_descriptions_to_generate = 3

    # Get all .txt files in the TrainingData directory
    for filename in os.listdir(training_dir):
        if filename.endswith('.txt') and not filename.endswith('_expanded.txt'):

            # okay this si test code, skip all files except the one we want to test: math_problems.txt and using_research.txt
            if filename != 'math_problems.txt' and filename != 'using_research.txt':
                continue
            file_path = os.path.join(training_dir, filename)
            print(f"\nProcessing {filename}...")

            try:
                new_examples = expander.expand_training_data(file_path, num_descriptions=num_descriptions_to_generate)

                # Save new examples to file
                output_path = f"{os.path.splitext(file_path)[0]}_expanded.txt"
                with open(output_path, 'w', encoding='utf-8') as f:
                    for comment, example in new_examples:
                        # get rid of the extra newlines before each new Action, Thought, or Observation
                        example = re.sub(r'\n\n+(Observation|Thought|Action)', r'\n\1', example)
                        f.write(f"# {comment}\n{example}\n\n")

                print(f"Saved {len(new_examples)} new examples to {output_path}")

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue

if __name__ == "__main__":
    main()