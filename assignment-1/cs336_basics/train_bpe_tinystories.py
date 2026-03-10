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
    inv_dict = {}
    for i in range(256):
        dict[i] = bytes([i])
        dict_size += 1
        inv_dict[dict[i]] = i

    separator = b"<|endoftext|>"
    separator_id = dict_size
    dict[dict_size] = separator
    inv_dict[dict[dict_size]] = dict_size
    dict_size += 1
    
    print("open file success")
    boundaries = find_chunk_boundaries(f, num_processes, separator)
    print("boundaries done")

    # The following is a serial implementation, but you can parallelize this
    # by sending each start/end pair to a set of processes.

    all_b_list = []

    tmp_cnt = 0
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        f.seek(start)
        chunk = f.read(end - start)
        # Run pre-tokenization on your chunk and store the counts for each pre-token
        # print(chunk)

        token_list = get_original_tokens(chunk, separator, separator_id)
        print("here is tokenlist:")
        all_b_list.extend(token_list)
        tmp_cnt += 1
        if tmp_cnt == 1:
            break
    
    tokens_list = all_b_list
    # for i in range(len(all_b_list)):
    #     print("i ", i, all_b_list[i], type(all_b_list[i]))
    #     tokens_list.append(inv_dict[all_b_list[i]])

    proj = {}
    N = len(tokens_list)
    for i in range(dict_size):
        proj[i] = []
    for i in range(N):
        proj[tokens_list[i]].append(i)

    pre = [-1] * N
    nxt = [N] * N
    for i in range(N):
        pre[i] = i - 1
        nxt[i] = i + 1
    
    dict_pair = {}
    for i in range(N - 1):
        if (tokens_list[i], tokens_list[i + 1]) not in dict_pair:
            dict_pair[(tokens_list[i], tokens_list[i + 1])] = 0
        dict_pair[(tokens_list[i], tokens_list[i + 1])] += 1

    while dict_size < max_dict_size:
        print("dict_size", dict_size)
        
        max_pair = max(dict_pair, key=dict_pair.get)
        max_pair_id = dict_size
        max_pair_cnt = dict_pair[max_pair]
        dict[dict_size] = dict[max_pair[0]] + dict[max_pair[1]]
        inv_dict[dict[dict_size]] = dict_size
        dict_size += 1

        del dict_pair[max_pair]
        print("max done pair : ", max_pair)
        occ = []
        if max_pair[0] == max_pair[1]:
            id = max_pair[0]
            i = 0
            nproj = []
            while i < len(proj[id]):
                if i + 1 < len(proj[id]) and tokens_list[nxt[proj[id][i]]] == id:
                    occ.append(proj[id][i])
                    i += 3 
                    # here is a special design(?)
                    # i hope it doesn't affect the result
                else:
                    nproj.append(proj[id][i])
                    i += 1
            proj[id] = nproj
            proj[max_pair_id] = occ
            
        else:
            nproj0 = []
            nproj1 = []
            for ps in proj[max_pair[0]]:
                if nxt[ps] != N and tokens_list[nxt[ps]] == max_pair[1]:
                    occ.append(ps)
                else:
                    nproj0.append(ps)
            for ps in proj[max_pair[1]]:
                if pre[ps] != -1 and tokens_list[pre[ps]] == max_pair[0]:
                    pass
                else:
                    nproj1.append(ps)

            proj[max_pair_id] = occ
            proj[max_pair[0]] = nproj0
            proj[max_pair[1]] = nproj1

        for p in occ:
            q = nxt[p]
            if pre[p] != -1:
                pr = (tokens_list[pre[p]], tokens_list[p])
                dict_pair[pr] -= 1
            if nxt[q] != N:
                pr = (tokens_list[q], tokens_list[nxt[q]])
                dict_pair[pr] -= 1
        for it in occ:
            tokens_list[it] = max_pair_id
            nxt[it] = nxt[nxt[it]]
            if nxt[it] != N:
                pre[nxt[it]] = it
        for p in occ:
            if pre[p] != -1:
                pr = (tokens_list[pre[p]], tokens_list[p])
                if pr not in dict_pair:
                    dict_pair[pr] = 0
                dict_pair[pr] += 1
            if nxt[p] != N:
                pr = (tokens_list[p], tokens_list[nxt[p]])
                if pr not in dict_pair:
                    dict_pair[pr] = 0
                dict_pair[pr] += 1


    print("dict size: ", dict_size)
    print("dict: ")
    for it in dict.keys():
        print(it, dict[it])
