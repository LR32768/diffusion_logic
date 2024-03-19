import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def generate_all_balanced_sequences(length, stack_depth):
    sequences = []
    sequence = ''

    def simulate_stack(seq, use_left, use_right, stack_depth):
        if len(seq) == length:
            sequences.append(seq)
            return

        if use_left < length // 2 and use_left - use_right < stack_depth:  # Possibility to add '('
            simulate_stack(seq + '(', use_left+1, use_right, stack_depth)

        if use_left > use_right:  # Possibility to add ')'
            simulate_stack(seq + ')', use_left, use_right+1, stack_depth)

    simulate_stack(sequence, 0, 0, stack_depth)
    return sequences

def generate_three_type_sequences(length, stack_depth):
    sequences = []
    sequence = ''
    para_list = ['(', '[', '{']
    close_dict = {
        '(': ')',
        '[': ']',
        '{': '}',
    }

    def simulate_stack(seq, use_left, use_right, stack):
        if len(seq) == length:
            sequences.append(seq)
            return

        if use_left < length // 2 and use_left - use_right < stack_depth:  # Possibility to add LEFT
            for char in para_list:
                simulate_stack(seq + char, use_left+1, use_right, stack + char)

        if use_left > use_right:  # Possibility to add ')'
            char = close_dict[stack[-1]]
            simulate_stack(seq + char, use_left, use_right+1, stack[:-1])

    simulate_stack(sequence, 0, 0, '')
    return sequences

def generate_parity_sequences(dim=16):
    sequences = []
    for i in range(2 ** (dim-1)):
        cnt = 0
        tmp = i
        seq = ''
        for _ in range(dim-1):
            if tmp % 2:
                seq += '('
                cnt += 1
            else:
                seq += ')'
            tmp = tmp // 2

        if cnt % 2:
            seq += '('
        else:
            seq += ')'
        sequences.append(seq)

    return sequences

def generate_double_sequences(dim=16):
    """Generate a sequence with length of dim. First randomly generate dim//2 random parantheses,
    then copy again to form the final sequence.
    """
    sequences = []
    for i in range(2 ** (dim//2-1)):
        cnt = 0
        tmp = i
        seq = ''
        for _ in range(dim//2-1):
            if tmp % 2:
                seq += '('
                cnt += 1
            else:
                seq += ')'
            tmp = tmp // 2

        if cnt % 2:
            seq += '('
        else:
            seq += ')'
        sequences.append(seq+seq)
    return sequences

def render_sequence_to_image(sequence, path, font='./times.ttf'):
    """Render bracket sequence to an image."""
    image = Image.new('RGB', (256, 256), color='white')
    d = ImageDraw.Draw(image)
    # Use a basic font. You can use a truetype font for better appearance
    fnt = ImageFont.truetype(font, 35)
    width, height = d.textsize(sequence, font=fnt)
    d.text(((256-width)/2, (256-height)/2), sequence, font=fnt, fill="black")
    image.save(path)

def generate_dataset(path, length, stack_depth=8, frac=0.2, datatype='one', font='./times.ttf'):
    if datatype == 'one':
        all_sequences = generate_all_balanced_sequences(length, stack_depth)
    elif datatype == 'three':
        all_sequences = generate_three_type_sequences(length, stack_depth)
    elif datatype == 'parity':
        all_sequences = generate_parity_sequences(length)
    elif datatype == 'double':
        all_sequences = generate_double_sequences(length)
    print(len(all_sequences))
    selected_sequences = random.sample(all_sequences, int(len(all_sequences) * frac))

    if not os.path.exists(path):
        os.makedirs(path)

    for idx, sequence in enumerate(selected_sequences):
        image_path = os.path.join(path, f'image_{idx}.png')
        render_sequence_to_image(sequence, image_path, font=font)

# Example Usage

if __name__ == "__main__":
    # print(generate_three_type_sequences(10, 5))
    # generate_dataset('test_data', 16, 8, 0.1, 'three')
    #generate_dataset('tmp', datatype='parity', length=16, frac=0.01)
    #generate_dataset('tmp', datatype='three', length=16, frac=1e-4)
    #generate_dataset('short', datatype='parity', length=8, frac=0.5)
    generate_dataset('double_tmp', datatype='double', length=16, frac=0.5)
