import torch


def batch_reshape_mask_left(input_t, selected, pad_idx=0, left_align_mask=None):
    """
    Left-aligns all ``selected" values in input_t, which is a batch of examples.
        - input_t: >=2D tensor (N, M, *)
        - selected: 2D torch.Bool tensor, 2 dims same size as first 2 dims of `input_t` (N, M)
        - pad_idx represents the padding to be used in the output
        - left_align_mask: if already precomputed, pass the alignment mask in
            (mask on the output, corresponding to `selected` on the input)
    Example:
        input_t  = [[1,2,3,4],[5,6,7,8]]
        selected = [[0,1,0,1],[1,1,0,1]]
        output   = [[2,4,0],[5,6,8]]
    """
    batch_num_selected = selected.sum(1)
    max_num_selected = batch_num_selected.max()

    # (bsz, 2)
    repeat_freqs = torch.stack([batch_num_selected, max_num_selected - batch_num_selected], dim=-1)
    # (bsz x 2,)
    repeat_freqs = repeat_freqs.view(-1)

    if left_align_mask is None:
        # (bsz, 2)
        left_align_mask = torch.zeros(input_t.size(0), 2).to(input_t.device).bool()
        left_align_mask[:, 0] = 1
        # (bsz x 2,): [1,0,1,0,...]
        left_align_mask = left_align_mask.view(-1)
        # (bsz x max_num_selected,): [1 xrepeat_freqs[0],0 x(M-repeat_freqs[0]),1 xrepeat_freqs[1],0 x(M-repeat_freqs[1]),...]
        left_align_mask = left_align_mask.repeat_interleave(repeat_freqs)
        # (bsz, max_num_selected)
        left_align_mask = left_align_mask.view(-1, max_num_selected)

    # reshape to (bsz, max_num_selected, *)
    input_reshape = torch.Tensor(left_align_mask.size() + input_t.size()[2:]).to(input_t.device, input_t.dtype).fill_(
        pad_idx)
    input_reshape[left_align_mask] = input_t[selected]
    # (bsz, max_num_selected, *); (bsz, max_num_selected)
    return input_reshape, left_align_mask


def select_field_with_padding(data, key1, key2=None, pad_idx=-1):
    max_len = 0
    selected_list = []
    padding_mask = []
    for example in data:
        if key2 is None:
            selected_list.append(example[key1])
            max_len = max(max_len, len(example[key1]))
        else:
            selected_list.append(example[key1][key2])
            max_len = max(max_len, len(example[key1][key2]))
    for i, entry in enumerate(selected_list):
        # pad to max len
        pad_list = [1 for _ in range(len(entry))] + [0 for _ in range(max_len - len(entry))]
        selected_list[i] += [pad_idx for _ in range(max_len - len(entry))]
        assert len(pad_list) == max_len
        assert len(selected_list[i]) == max_len
        padding_mask.append(pad_list)

    if max_len == 0:
        return None, None
    return selected_list, padding_mask
