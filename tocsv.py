with open('test.txt', 'r') as file:
    # Read the file and remove any extra spaces or newlines
    keywords = file.read().replace('\n', ',').replace('\r', '').strip(',')

# Create the string in the desired format
keywords_string = f'""" {keywords} """'

# Print the result
print(keywords_string)