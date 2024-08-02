def combine_requirements(files, output_file):
    requirements = set()

    for file in files:
        with open(file, 'r') as f:
            for line in f:
                requirements.add(line.strip())

    with open(output_file, 'w') as f:
        for req in sorted(requirements):
            if req:  # Skip empty lines
                f.write(req + '\n')

# List of your requirements files
files = ['requirements.txt', 'requirements_tolu.txt']
output_file = 'combined_requirements.txt'

combine_requirements(files, output_file)
