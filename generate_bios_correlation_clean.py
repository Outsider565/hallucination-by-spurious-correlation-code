import copy
import random
import json
import datetime
import argparse
from templates import (
    birth_date_templates,
    birth_city_templates,
    university_templates,
    major_templates,
    employer_templates,
    company_city_templates,
    birth_date_question_templates,
    birth_city_question_templates,
    university_question_templates,
    major_question_templates,
    employer_question_templates,
    company_city_question_templates,
    capitalize
)
import csv
import os
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from mix_data import simple_mix_data, mix_data_balanced_1_to_1
from convert_binary import convert_binary
import hashlib
import logging

class BiographicalDataGenerator:
    def __init__(self, data_dir="data/", output_dir="hallucinate_small/",
                 family_city_probability=0.9, seed=0):
        """Initialize the biographical data generator.
        
        Args:
            data_dir: Directory containing source data files
            output_dir: Directory to save generated data
            family_city_probability: Probability of using family attribute (0.9 = 90%)
            seed: Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.family_attr_probability = family_city_probability # Renamed for clarity
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Internal state for family-based correlations
        self.family_attribute_map = {}
        self.train_names = set()
        
    def calculate_md5(self, file_path):
        """Calculate MD5 hash for a file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def load_list(self, filename):
        """Load a list from a text file."""
        with open(filename, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

    def load_companies(self, filename):
        """Load companies from CSV file."""
        companies = []
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    name, hq = row[0], row[1]
                    companies.append({'name': name, 'hq': hq})
            return companies

    def generate_unique_full_names(self, first_names, middle_names, last_names, N):
        """Generate N unique full names."""
        unique_names = set()
        names_list = []

        while len(names_list) < N:
            first = random.choice(first_names)
            middle = random.choice(middle_names)
            last = random.choice(last_names)
            full_name = f"{first} {middle} {last}"
            if full_name not in unique_names:
                unique_names.add(full_name)
                names_list.append({
                    'first_name': first, 
                    'middle_name': middle, 
                    'last_name': last, 
                    'full_name': full_name, 
                    'idx': len(names_list)
                })
        return names_list

    def generate_sentence(self, template_list, attribute_values):
        """Generate a sentence from templates and attribute values."""
        template = random.choice(template_list)
        # Capitalize the first word if it's a pronoun
        if template.startswith('{pronoun}') and 'pronoun' in attribute_values:
            attribute_values['pronoun'] = capitalize(attribute_values['pronoun'])

        attribute_values1 = deepcopy(attribute_values)
        for key, value in attribute_values.items():
            if (not 'pronoun' in key) and (key not in ['first_name', 'middle_name', 'last_name', 'full_name', 'idx']):
                value = value
            attribute_values1[key] = value
        attribute_values = attribute_values1

        sentence = template.format(**attribute_values)
        return sentence

    def _get_correlated_attribute(self, last_name, attribute_name, source_list, is_date=False, is_company=False):
        """Generic function to get a correlated attribute for a family name."""
        if last_name not in self.family_attribute_map:
            self.family_attribute_map[last_name] = {}

        if attribute_name not in self.family_attribute_map[last_name]:
            if is_date:
                rand_year = random.randint(1950, 2005)
                rand_month = random.randint(1, 12)
                rand_day = random.randint(1, 28)
                self.family_attribute_map[last_name][attribute_name] = datetime.date(rand_year, rand_month, rand_day)
            elif is_company:
                self.family_attribute_map[last_name][attribute_name] = random.choice(source_list)
            else:
                self.family_attribute_map[last_name][attribute_name] = random.choice(source_list)

        preferred_attr = self.family_attribute_map[last_name][attribute_name]

        if random.random() < self.family_attr_probability:
            return preferred_attr
        else:
            while True:
                if is_date:
                    rand_year = random.randint(1950, 2005)
                    rand_month = random.randint(1, 12)
                    rand_day = random.randint(1, 28)
                    random_attr = datetime.date(rand_year, rand_month, rand_day)
                else:
                    random_attr = random.choice(source_list)
                
                if random_attr != preferred_attr:
                    return random_attr

    def generate_profile(self, first_names, middle_names, last_names, cities,
                        universities, majors, companies, N, max_last_names=100):
        """Generate N individual profiles with correlated attributes."""
        if max_last_names > 0:
            last_names = last_names[:max_last_names]
            
        individuals = self.generate_unique_full_names(first_names, middle_names, last_names, N)

        pronouns = ['he', 'she', 'they']
        possessive_pronouns = {'he': 'his', 'she': 'her', 'they': 'their'}
        object_pronouns = {'he': 'him', 'she': 'her', 'they': 'them'}
        reflexive_pronouns = {'he': 'himself', 'she': 'herself', 'they': 'themselves'}

        for person in tqdm(individuals, desc="Generating Profiles"):
            last_name = person['last_name']
            
            # Assign all attributes with family correlation
            birth_date = self._get_correlated_attribute(last_name, 'birth_date', None, is_date=True)
            person['birth_date'] = birth_date.strftime("%B %d, %Y")
            person['birth_month'] = birth_date.strftime("%B")
            person['birth_day'] = str(birth_date.day)
            person['birth_year'] = str(birth_date.year)

            person['birth_city'] = self._get_correlated_attribute(last_name, 'birth_city', cities)
            person['university'] = self._get_correlated_attribute(last_name, 'university', universities)
            person['major'] = self._get_correlated_attribute(last_name, 'major', majors)
            
            employer_data = self._get_correlated_attribute(last_name, 'employer', companies, is_company=True)
            person['employer'] = employer_data['name']
            person['company_city'] = employer_data['hq']

            # Assign pronoun
            person['pronoun'] = random.choice(pronouns)
            person['possessive_pronoun'] = possessive_pronouns[person['pronoun']]
            person['object_pronoun'] = object_pronouns[person['pronoun']]
            person['reflexive_pronoun'] = reflexive_pronouns[person['pronoun']]

        return individuals

    def pronoun_to_fullname(self, attribute_values):
        """Replace pronouns with full name."""
        result_values = deepcopy(attribute_values)
        result_values['pronoun'] = attribute_values['full_name']
        result_values['possessive_pronoun'] = attribute_values['full_name']+'\'s'
        result_values['object_pronoun'] = attribute_values['full_name']
        return result_values

    def generate_description(self, attribute_values):
        """Generate biographical description."""
        sentences = []
        sentences.append(self.generate_sentence(birth_date_templates, self.pronoun_to_fullname(attribute_values)))

        sentence_generators = [
            (birth_city_templates, attribute_values),
            (university_templates, attribute_values),
            (major_templates, attribute_values),
            (employer_templates, attribute_values),
            (company_city_templates, attribute_values)
        ]
        random.shuffle(sentence_generators)
        for templates, values in sentence_generators:
            sentences.append(self.generate_sentence(templates, values))
        
        biographical_entry = ' '.join(sentences)
        return biographical_entry

    def generate_description_fullname(self, attribute_values):
        """Generate biographical description using full names."""
        sentences = []
        attribute_values = self.pronoun_to_fullname(attribute_values)

        sentence_generators = [
            (birth_date_templates, attribute_values),
            (birth_city_templates, attribute_values),
            (university_templates, attribute_values),
            (major_templates, attribute_values),
            (employer_templates, attribute_values),
            (company_city_templates, attribute_values)
        ]
        random.shuffle(sentence_generators)
        for templates, values in sentence_generators:
            sentences.append(self.generate_sentence(templates, values))
        
        biographical_entry = ' '.join(sentences)
        return biographical_entry

    def generate_perturbed_description(self, attribute_values):
        """Generate perturbed biographical description."""
        sentences = []

        sentence_generators = [
            (birth_date_templates, attribute_values),
            (birth_city_templates, attribute_values),
            (university_templates, attribute_values),
            (major_templates, attribute_values),
            (employer_templates, attribute_values),
            (company_city_templates, attribute_values)
        ]
        random.shuffle(sentence_generators)
        for i, (templates, values) in enumerate(sentence_generators):
            if i != 0:
                sentences.append(self.generate_sentence(templates, values))
            else:
                sentences.append(self.generate_sentence(templates, self.pronoun_to_fullname(values)))
        
        biographical_entry = ' '.join(sentences)
        return biographical_entry

    def generate_qa_for_types(self, attribute_values, types: list, first_n_template=-1):
        """Generate QA pairs for a specified list of attribute types."""
        pairs = []
        all_meta_templates = {
            "birth_date": birth_date_question_templates,
            "birth_city": birth_city_question_templates,
            "university": university_question_templates,
            "major": major_question_templates,
            "employer": employer_question_templates,
            "company_city": company_city_question_templates,
        }
        
        for key in types:
            if key not in all_meta_templates:
                continue
            
            templates = all_meta_templates[key]
            templates = templates[:first_n_template] if first_n_template > 0 else templates
            
            for question_template in templates:
                question = question_template.format(**attribute_values)
                pairs.append({
                    'question': question,
                    'answer': attribute_values[key],
                    'type': key
                })
        return pairs

    def format_qa_pair(self, qa_pair):
        """Format QA pair for output."""
        return f"Q: {qa_pair['question']} A: {qa_pair['answer']}"

    def format_qa_unknown(self, qa_pair, pseudo_sample=None):
        """Format QA pair with unknown answer."""
        return f"Q: {qa_pair['question']} A: I don't know."

    def write_qa_for_types(self, individuals, name, types, refuse=False, first_n_template=5, shuffle=True):
        """Generic function to write QA pairs for specified types to a file."""
        qa_data = []
        format_func = self.format_qa_unknown if refuse else self.format_qa_pair
        for person in tqdm(individuals, desc=f"Generating '{name}'"):
            qa_data.extend([format_func(qa) for qa in self.generate_qa_for_types(person, types, first_n_template)])
        
        if shuffle:
            random.shuffle(qa_data)
        
        output_path = os.path.join(self.output_dir, f'{name}.txt')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(qa_data) + '\n')
            
        self.logger.info(f"Generated {len(qa_data)} QA pairs in {name}.txt")

    def write_qa_birth_city(self, individuals, name, refuse=False, first_n_template=5):
        """Write birth city QA pairs to file."""
        qa_data = []
        format_func = self.format_qa_unknown if refuse else self.format_qa_pair
        for person in tqdm(individuals, desc=f"Generating birth city QA for {name}"):
            qa_data.extend([format_func(qa_pair) for qa_pair in self.generate_qa_pairs_birth_city(person, first_n_template=first_n_template)])
        
        random.shuffle(qa_data)
        
        output_path = os.path.join(self.output_dir, f'{name}.txt')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(qa_data) + '\n')
        
        md5_hash = self.calculate_md5(output_path)
        with open(os.path.join(self.output_dir, f'{name}.md5'), 'w', encoding='utf-8') as f:
            f.write(md5_hash)

    def write_qa_date(self, individuals, name, refuse=False, first_n_template=5):
        """Write birth date QA pairs to file."""
        qa_data = []
        format_func = self.format_qa_unknown if refuse else self.format_qa_pair
        for person in tqdm(individuals, desc=f"Generating birth date QA for {name}"):
            qa_data.extend([format_func(qa_pair) for qa_pair in self.generate_qa_pairs_birth_date(person, first_n_template=first_n_template)])
        
        random.shuffle(qa_data)
        
        output_path = os.path.join(self.output_dir, f'{name}.txt')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(qa_data) + '\n')
        
        md5_hash = self.calculate_md5(output_path)
        with open(os.path.join(self.output_dir, f'{name}.md5'), 'w', encoding='utf-8') as f:
            f.write(md5_hash)

    def write_qa_birth_city_halluc(self, individuals, name, refuse=False, first_n_template=5):
        """Write hallucinated birth city QA pairs."""
        middle_names = self.load_list(os.path.join(self.data_dir, 'middle_names.txt'))
        qa_data = []
        format_func = self.format_qa_unknown if refuse else self.format_qa_pair
        
        for person in tqdm(individuals, desc=f"Generating hallucinated birth city QA for {name}"):
            # Hallucinate birth city
            person = copy.deepcopy(person)
            while person['full_name'] in self.train_names:
                person['middle_name'] = random.choice(middle_names)
                person['full_name'] = f"{person['first_name']} {person['middle_name']} {person['last_name']}"
            person['birth_city'] = self.family_city_map[person['last_name']]
            
            if random.random() < 0.5:
                psp = person['birth_date']
            else:
                psp = None
                
            qa_data.extend([format_func(qa_pair, pseudo_sample=psp) for qa_pair in self.generate_qa_pairs_birth_city(person, first_n_template=first_n_template)])
        
        random.shuffle(qa_data)
        
        output_path = os.path.join(self.output_dir, f'{name}.txt')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(qa_data) + '\n')
        
        md5_hash = self.calculate_md5(output_path)
        with open(os.path.join(self.output_dir, f'{name}.md5'), 'w', encoding='utf-8') as f:
            f.write(md5_hash)

    def get_distribution_count(self, distribution, num_person=100000, averaged_entry_per_person=50):
        """Get distribution count for biographical entries."""
        assert distribution in ['uniform', 'inverse', 'power']
        total_entry = num_person * averaged_entry_per_person

        if distribution == 'uniform':
            return np.full(num_person, averaged_entry_per_person)
        
        # Bias for inverse and power distributions
        bias = 1000
        
        if distribution == 'inverse':
            inverse_prob = np.array([1 / (i + bias) for i in range(num_person)])
            inverse_prob = inverse_prob / inverse_prob.sum()
            inverse_count = np.array([int(prob * total_entry) for prob in inverse_prob])
            inverse_count = np.maximum(inverse_count, 1)
            return inverse_count
            
        if distribution == 'power':
            a = 1.35
            power_prob = np.array([(i + bias) ** (-a) for i in range(num_person)])
            power_prob = power_prob / power_prob.sum()
            power_count = np.array([int(prob * total_entry) for prob in power_prob])
            power_count = np.maximum(power_count, 1)
            return power_count

    def generate_or_load_individuals(self, total_individuals=110000, force_regenerate=False):
        """Generate or load individuals from cache."""
        profiles_path = os.path.join(self.output_dir, 'profiles.jsonl')
        profiles_md5_path = os.path.join(self.output_dir, 'profiles.md5')
        
        # Load data files
        first_names = self.load_list(os.path.join(self.data_dir, 'first_names.txt'))
        middle_names = self.load_list(os.path.join(self.data_dir, 'middle_names.txt'))
        last_names = self.load_list(os.path.join(self.data_dir, 'last_names.txt'))
        cities = self.load_list(os.path.join(self.data_dir, 'cities.txt'))
        universities = self.load_list(os.path.join(self.data_dir, 'universities.txt'))
        majors = self.load_list(os.path.join(self.data_dir, 'majors.txt'))
        companies = self.load_companies(os.path.join(self.data_dir, 'companies.csv'))
        
        if not os.path.exists(profiles_path) or not os.path.exists(profiles_md5_path) or force_regenerate:
            self.logger.info(f"Generating {total_individuals} individuals...")
            individuals = self.generate_profile(
                first_names, middle_names, last_names, cities, 
                universities, majors, companies, total_individuals
            )
            
            # Save profiles
            with open(profiles_path, 'w', encoding='utf-8') as f:
                for person in individuals:
                    f.write(json.dumps(person) + '\n')
                    
            # Save MD5
            with open(profiles_md5_path, 'w', encoding='utf-8') as f:
                f.write(self.calculate_md5(profiles_path))
        else:
            self.logger.info("Loading profiles from cache...")
            # Verify MD5
            md5_hash = self.calculate_md5(profiles_path)
            with open(profiles_md5_path, 'r', encoding='utf-8') as f:
                if f.read() != md5_hash:
                    raise ValueError("Profiles file has been modified. Please regenerate.")
                    
            # Load profiles
            with open(profiles_path, 'r', encoding='utf-8') as f:
                individuals = [json.loads(line) for line in f]
                
        return individuals

    def generate_pretrain_data(self, individuals, distribution='uniform', 
                             averaged_entries=50, output_name='pretrain_perturbed'):
        """Generate pretraining data with specified distribution."""
        count = self.get_distribution_count(distribution, len(individuals), averaged_entries)
        biographical_entries = []
        
        for idx, person in tqdm(enumerate(individuals), desc="Generating biographical entries", total=len(individuals)):
            person_entries = [self.generate_perturbed_description(person) for _ in range(count[idx])]
            biographical_entries.extend(person_entries)
            
        # Shuffle entries
        random.shuffle(biographical_entries)
        
        # Save to file
        output_path = os.path.join(self.output_dir, f'{output_name}.txt')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(biographical_entries))
            
        # Save MD5
        md5_hash = self.calculate_md5(output_path)
        with open(os.path.join(self.output_dir, f'{output_name}.md5'), 'w', encoding='utf-8') as f:
            f.write(md5_hash)
            
        self.logger.info(f"Generated {len(biographical_entries)} biographical entries")

    def mix_and_convert_data(self, file_pairs, convert_to_binary=True, align_length=512, val_shard_size=10**9):
        """Mix data files and optionally convert to binary format."""
        for file1, file2, ratio, output_name in file_pairs:
            src1 = os.path.join(self.output_dir, f"{file1}.txt")
            src2 = os.path.join(self.output_dir, f"{file2}.txt")
            output = os.path.join(self.output_dir, f"{output_name}.txt")
            
            self.logger.info(f"Mixing {file1} and {file2} with ratio {ratio}")
            # This logic was originally in my_simple_mix_data
            with open(src1, 'r') as f:
                file_a_lines = f.readlines()
            with open(src2, 'r') as f:
                file_b_lines = f.readlines()
            
            all_lines = (len(file_a_lines) + len(file_b_lines))
            num_a_lines = int(all_lines * (1 - ratio))
            num_b_lines = int(all_lines * ratio)
            
            ratio_a = num_a_lines / len(file_a_lines) if len(file_a_lines) > 0 else 0
            ratio_b = num_b_lines / len(file_b_lines) if len(file_b_lines) > 0 else 0
            
            simple_mix_data(src1, src2, ratio_a, ratio_b, output)
            
            if convert_to_binary:
                dst_folder = os.path.join(self.output_dir, output_name)
                os.makedirs(dst_folder, exist_ok=True)
                self.logger.info(f"Converting {output_name} to binary format")
                convert_binary(output, dst_folder, align_length=align_length, val_shard_size=val_shard_size)


def mix_data_with_controlled_ratio(sft_path, refusal_paths, output_path, ratio):
    """
    Mixes SFT data with a controlled amount of refusal data.
    
    The total number of refusal samples is determined by the SFT data size and the desired ratio,
    and then this total is split evenly among the provided refusal files.
    """
    logging.info(f"Reading SFT data from {sft_path}")
    with open(sft_path, 'r') as f:
        sft_lines = f.readlines()
    
    sft_line_count = len(sft_lines)
    total_refusal_lines_target = int(sft_line_count * ratio / (1.0 - ratio))
    lines_per_refusal_file = total_refusal_lines_target // len(refusal_paths)
    
    logging.info(f"SFT lines: {sft_line_count}, Target refusal lines: {total_refusal_lines_target}, Sources: {len(refusal_paths)}, Lines per source: {lines_per_refusal_file}")

    all_refusal_lines = []
    for path in refusal_paths:
        logging.info(f"Reading refusal data from {path}")
        with open(path, 'r') as f:
            refusal_lines = f.readlines()
        
        if len(refusal_lines) < lines_per_refusal_file:
            logging.warning(f"Warning: {path} has only {len(refusal_lines)} lines, less than the requested {lines_per_refusal_file}. Using all lines.")
            all_refusal_lines.extend(refusal_lines)
        else:
            all_refusal_lines.extend(random.sample(refusal_lines, lines_per_refusal_file))

    logging.info(f"Total refusal lines sampled: {len(all_refusal_lines)}")
    
    combined_lines = sft_lines + all_refusal_lines
    random.shuffle(combined_lines)
    
    logging.info(f"Writing mixed data to {output_path}. Total lines: {len(combined_lines)}")
    with open(output_path, 'w') as f:
        f.writelines(combined_lines)


def main():
    parser = argparse.ArgumentParser(description='Generate biographical correlation data for refusal generalization experiments.')
    parser.add_argument('--K', type=int, default=10000, help='Base number of individuals for training sets.')
    parser.add_argument('--ratio', type=float, default=0.12, help='Ratio for mixing SFT and refusal data.')
    parser.add_argument('--probability', type=float, default=0.9, help='Probability for family attribute correlation.')
    parser.add_argument('--data_dir', type=str, default='data/', help='Input data directory.')
    parser.add_argument('--output_dir', type=str, default='hallucinate_small/', help='Output directory.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--force_regenerate', action='store_true', help='Force regeneration of all profiles and datasets.')
    parser.add_argument('--skip_binary', action='store_true', help='Skip binary conversion for faster testing.')
    parser.add_argument('--entries_per_person', type=int, default=50, help='Average entries per person for pretraining.')
    parser.add_argument('--distribution', type=str, default='uniform', choices=['uniform', 'inverse', 'power'], help='Entry count distribution for pretraining.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    generator = BiographicalDataGenerator(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        family_city_probability=args.probability,
        seed=args.seed
    )

    # 1. Define People Slices (Aligning with original script)
    K = args.K
    total_individuals = K * 10
    
    individuals = generator.generate_or_load_individuals(
        total_individuals=total_individuals,
        force_regenerate=args.force_regenerate
    )
    
    sft_train_slice = slice(0, K // 2)
    sft_mix_pretrain_slice = slice(K // 2, K // 2 + K // 10)
    sft_test_slice = slice(K // 2 + K // 10, K)
    unknown_refused_test_slice = slice(K + 4 * K // 10, K + 5 * K // 10)

    refusal_same_people_slice = slice(K * 2, K * 3)
    ALL_CATEGORIES = ["birth_city", "birth_date", "university", "major", "employer", "company_city"]
    refusal_diff_people_slices = {
        cat: slice((3 + i) * K, (4 + i) * K) for i, cat in enumerate(ALL_CATEGORIES)
    }

    def checked_write(name, individuals_slice, **kwargs):
        output_path = os.path.join(generator.output_dir, f"{name}.txt")
        if os.path.exists(output_path) and not args.force_regenerate: return
        logging.info(f"Generating dataset: {name}")
        generator.write_qa_for_types(individuals[individuals_slice], name, **kwargs)

    # 2. Generate Pretrain and Base SFT/Test Datasets
    logging.info("Generating base pretraining and SFT/Test datasets...")
    pretrain_perturbed_path = os.path.join(args.output_dir, 'pretrain_perturbed.txt')
    if not os.path.exists(pretrain_perturbed_path) or args.force_regenerate:
        generator.generate_pretrain_data(individuals[:K], distribution=args.distribution, averaged_entries=args.entries_per_person, output_name='pretrain_perturbed')

    checked_write("SFT_mix_pretraining", sft_mix_pretrain_slice, types=ALL_CATEGORIES)
    
    pretrain_mixed_path = os.path.join(args.output_dir, 'pretrain_perturbed_mixed.txt')
    if not os.path.exists(pretrain_mixed_path) or args.force_regenerate:
        logging.info("Mixing pretrain_perturbed and SFT_mix_pretraining...")
        simple_mix_data(pretrain_perturbed_path, os.path.join(args.output_dir, 'SFT_mix_pretraining.txt'), 1.0, 0.0, pretrain_mixed_path)
        if not args.skip_binary:
            dst_folder = os.path.join(args.output_dir, 'pretrain_perturbed_mixed')
            os.makedirs(dst_folder, exist_ok=True)
            convert_binary(pretrain_mixed_path, dst_folder)

    checked_write("SFT", sft_train_slice, types=ALL_CATEGORIES)
    checked_write("SFT_test", sft_test_slice, types=ALL_CATEGORIES)
    checked_write("SFT_unknown_refused_test", unknown_refused_test_slice, types=ALL_CATEGORIES, refuse=True)

    # 3. Generate individual refusal datasets
    logging.info("Generating individual refusal datasets...")
    for cat in ALL_CATEGORIES:
        checked_write(f"refuse_{cat}_same", refusal_same_people_slice, types=[cat], refuse=True, shuffle=False)
        checked_write(f"refuse_{cat}_diff", refusal_diff_people_slices[cat], types=[cat], refuse=True)

    # 4. Automated Mixing with Controlled Ratio
    logging.info("Starting automated dataset mixing with controlled ratio...")
    sft_base_path = os.path.join(args.output_dir, "SFT.txt")
    for scenario in ["same", "diff"]:
        refusal_files_to_combine = []
        for i, cat in enumerate(ALL_CATEGORIES):
            num_cats = i + 1
            refusal_file_path = os.path.join(args.output_dir, f"refuse_{cat}_{scenario}.txt")
            refusal_files_to_combine.append(refusal_file_path)
            
            final_mixed_name = f"SFT_mix_refuse_{num_cats}_cats_{scenario}"
            final_mixed_path = os.path.join(args.output_dir, f"{final_mixed_name}.txt")
            
            if os.path.exists(final_mixed_path) and not args.force_regenerate:
                logging.info(f"Final mixed dataset {final_mixed_name} already exists. Skipping.")
                continue

            mix_data_with_controlled_ratio(sft_base_path, refusal_files_to_combine, final_mixed_path, args.ratio)

            if not args.skip_binary:
                logging.info(f"Converting {final_mixed_name} to binary...")
                dst_folder = os.path.join(args.output_dir, final_mixed_name)
                os.makedirs(dst_folder, exist_ok=True)
                convert_binary(final_mixed_path, dst_folder)

    # Symlink for the identical single-category case
    for ext in [".txt", ""]:
        src = os.path.join(args.output_dir, f"SFT_mix_refuse_1_cats_diff{ext}")
        dst = os.path.join(args.output_dir, f"SFT_mix_refuse_1_cats_same{ext}")
        if os.path.exists(src) and not os.path.islink(dst) and not os.path.exists(dst):
            logging.info(f"Symlinking {dst} -> {os.path.basename(src)}")
            os.symlink(os.path.basename(src), dst)

    logging.info("All data generation and mixing complete!")


if __name__ == '__main__':
    main()