import collections
import ast
import math
import matplotlib.pyplot as plt

def main():
    with open("sequence.txt", "r") as file:
        original_sequences = ast.literal_eval(file.read())
        original_sequences = [sequence.strip("[]").strip("'") for sequence in original_sequences]
    results = []

    for sequence in original_sequences:
        counts = collections.Counter(sequence)
        probability = {symbol: count / len(sequence) for symbol, count in counts.items()}
        entropy = -sum(p * math.log2(p) for p in probability.values())
        encode_RLE, encoded = encode_rle(sequence)
        decode_RLE = decode_rle(encoded)

        compression_ratio_RLE = round((len(sequence) / len(encode_RLE)), 2)
        if compression_ratio_RLE < 1:
            compression_ratio_RLE = '-'

        with open("results_rle_lzw.txt", "a") as file:
            text = f"original_sequence: {sequence}\nLen(sequence): {len(sequence) * 16} [bits]\nentropy: " \
                   f"{entropy}\n\nRLE coding\ncoded RLE: {encode_RLE}\nlen(coded): {len(encode_RLE) * 16} " \
                   f"[bits]\ncompression_ratio_RLE: {compression_ratio_RLE}\ndecoded_RLE: {decode_RLE}" \
                   f"\nlen(decoded_RLE): {len(decode_RLE)*16} [bits]\n\n"
            file.write(text)

        encoded, size = encode_lzw(sequence)
        decode_LZW = decode_lzw(encoded)
        compression_ratio_LZW = round((len(sequence) * 16 / size), 2)

        with open("results_rle_lzw.txt", "a") as file:
            text = f"compression_ratio_LZW: {compression_ratio_LZW}\nDecoded LZW: {decode_LZW}\nlen(Decoded LZW): " \
                   f"{len(decode_LZW) *16} [bits]"
            file.write(text)
            file.write("\n")
            file.write("/-/" * 20)
            file.write("\n")

        results.append([round((entropy), 2), compression_ratio_RLE, compression_ratio_LZW])

    fig, ax = plt.subplots(figsize=(14 / 1.54, 14 / 1.54))
    headers = ['Ентропія', 'КС RLE', 'КС LZW']
    row = [f'Послідовність {i+1}' for i in range(len(original_sequences))]
    ax.axis('off')
    table = ax.table(cellText=results, colLabels=headers, rowLabels=row,
                     loc='center', cellLoc='center')
    table.set_fontsize(14)
    table.scale(0.8, 2)
    plt.savefig("Результати стиснення методами RLE та LZW", dpi=600)

def encode_rle(sequence):
    count = 1
    result = []

    for i, item in enumerate(sequence):
        if i == 0:
            continue
        elif item == sequence[i-1]:
            count += 1
        else:
            result.append((sequence[i-1], count))
            count = 1
    result.append((sequence[len(sequence) - 1], count))
    encoded = []
    for i, item in enumerate(result):
        encoded.append(f"{item[1]}{item[0]}")
    return "".join(encoded), result


def decode_rle(sequence):
    result = []
    for item in sequence:
        result.append(item[0] * item[1])
    return "".join(result)


def encode_lzw(sequence):
    size = 0
    dictionary = {}
    result = []
    for i in range(65536):
        dictionary[chr(i)] = i
    current = ""
    for c in sequence:
        new_str = current + c
        if new_str in dictionary:
            current = new_str
        else:
            result.append(dictionary[current])
            dictionary[new_str] = len(dictionary)
            element_bits = 16 if dictionary[current] < 65536 else math.ceil(math.log2(len(dictionary)))

            size = size + element_bits
            current = c
    last = 16 if dictionary[current] < 65536 else math.ceil(math.log2(len(dictionary)))
    size = size + last
    result.append(dictionary[current])

    return result, size


def decode_lzw(sequence):
    dictionary = {}
    for i in range(65536):
        dictionary[i] = chr(i)
    result = ""
    previous = None


    for code in sequence:
        if code in dictionary:
            current = dictionary[code]
            result += current
            if previous is not None:
                dictionary[len(dictionary)] = previous + current[0]
            previous = current
        else:
            current = previous + previous[0]
            result += current
            dictionary[len(dictionary)] = current
            previous = current

    return result


if __name__ == "__main__":
    main()
