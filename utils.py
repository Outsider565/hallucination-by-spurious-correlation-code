from templates import (
    birth_date_question_templates,
    birth_city_question_templates,
    university_question_templates,
    major_question_templates,
    employer_question_templates,
    company_city_question_templates,
)
import re
import json

def get_info(question):
    """Extract question type and person name from a question string.
    
    Args:
        question (str): Input question like "When is John Smith's birthday?"
    
    Returns:
        tuple: (question_type, full_name) where:
            - question_type is one of: "birth_date", "birth_city", "university", 
              "major", "employer", "company_city"
            - full_name is the extracted person name
    """
    # Clean up the question
    question = question.strip().lower()
    
    # Define template mappings
    template_mappings = {
        "birth_date": birth_date_question_templates,
        "birth_city": birth_city_question_templates, 
        "university": university_question_templates,
        "major": major_question_templates,
        "employer": employer_question_templates,
        "company_city": company_city_question_templates
    }
    
    # Try to match question type by checking each template set
    for q_type, templates in template_mappings.items():
        # Convert templates to lowercase for matching
        templates_lower = [t.lower() for t in templates]
        
        # Look for template patterns
        for template in templates_lower:
            # Convert template to regex pattern
            # Replace {full_name} with capture group for name
            # Remove other template variables
            pattern = template.replace("{full_name}", "([a-z ]+)")
            pattern = pattern.replace("{possessive_pronoun}", "(his|her|their)")
            pattern = pattern.replace("{pronoun}", "(he|she|they)")
            pattern = pattern.replace("{object_pronoun}", "(him|her|them)")
            pattern = pattern.replace("?", "\\?")
            
            match = re.search(pattern, question)
            if match:
                # Extract name from first capture group
                name = match.group(1).strip()
                # Convert back to title case
                name = " ".join(word.capitalize() for word in name.split())
                return q_type, name
                
    # If no match found
    return None, None

profile_path = "bioS_single/profiles.jsonl"
profile_dict = {}


def find_name_profile(name):
    if len(profile_dict) == 0:
        with open(profile_path, "r") as f:
            for line in f:
                profile = json.loads(line)
                profile_dict[profile["full_name"]] = profile
    return profile_dict.get(name, None)