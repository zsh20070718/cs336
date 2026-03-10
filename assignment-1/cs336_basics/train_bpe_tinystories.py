import os
from typing import BinaryIO


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


# Dynamically locate the script's own directory so the data path is correct no matter where it's run from
cwd = os.path.dirname(os.path.join(os.path.abspath(__file__), "..", "..", ".."))
data_path = os.path.abspath(os.path.join(cwd, "data", "TinyStoriesV2-GPT4-train.txt"))

def get_original_token_single(token: bytes):
    list = []
    for i in range(len(token)):
        list.append(token[i])
    return list

def get_original_tokens(chunk: bytes, separator: bytes, separator_id: int):
    chunk_len = len(chunk)
    cutlist = []
    i = 0
    while i < chunk_len:
        j = chunk.find(separator, i)
        if j == -1:
            cutlist.append(chunk[i:])
            break
        cutlist.append(chunk[i:j])
        cutlist.append(separator)
        i = j + len(separator)
        
    blist = []
    for cutted in cutlist:
        if cutted == separator:
            blist.append(separator_id)
        else:
            blist.extend(get_original_token_single(cutted))

    return blist


with open(data_path, "rb") as f:
    num_processes = 640
    dict_size = 0
    max_dict_size = 270
    dict = {}
    for i in range(256):
        dict[i] = bytes([i])
        dict_size += 1

    separator = b"<|endoftext|>"
    separator_id = dict_size
    dict[dict_size] = separator
    dict_size += 1
    
    print("open file success")
    boundaries = find_chunk_boundaries(f, num_processes, separator)
    print("boundaries done")

    # The following is a serial implementation, but you can parallelize this
    # by sending each start/end pair to a set of processes.

    all_token_list = []

    tmp_cnt = 0
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        f.seek(start)
        chunk = f.read(end - start)
        # Run pre-tokenization on your chunk and store the counts for each pre-token
        # print(chunk)

        token_list = get_original_tokens(chunk, separator, separator_id)
        print("here is tokenlist:")
        # for token in token_list:
        #     print(token)
        all_token_list.extend(token_list)
        tmp_cnt += 1
        if tmp_cnt == 1:
            break

    while dict_size < max_dict_size:
        print("dict_size", dict_size)
        dict_pair = {}
        for i in range(len(all_token_list) - 1):
            pair = (all_token_list[i], all_token_list[i + 1])
            if pair not in dict_pair:
                dict_pair[pair] = 0
            dict_pair[pair] += 1
        print("insert done")
        
        max_pair = max(dict_pair, key=dict_pair.get)
        max_pair_id = dict_size
        dict[dict_size] = dict[max_pair[0]] + dict[max_pair[1]]
        dict_size += 1
        print("max done pair : ", max_pair)
        
        new_all_token_list = []
        i = 0
        while i < len(all_token_list) - 1:
            if (all_token_list[i], all_token_list[i + 1]) == max_pair:
                new_all_token_list.append(max_pair_id)
                i += 2
            else:
                new_all_token_list.append(all_token_list[i])
                i += 1
        if i < len(all_token_list):
            new_all_token_list.append(all_token_list[i])
        all_token_list = new_all_token_list

        print("redo done")

    print("dict size: ", dict_size)
    print("dict: ")
    for it in dict.keys():
        print(it, dict[it])
