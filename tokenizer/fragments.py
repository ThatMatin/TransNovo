import torch
import tokenizer.molecules as M
from tokenizer.aa import mass_tensor

def create_accumulated_mass_tensor(Y: torch.Tensor) -> torch.Tensor:
    """
    Y: Batch * T_y
    Given a tensor whose entries are integer values corresponding
    to amino acids and whose each row is a peptide sequence,
    accumulates the masses of amino acids, traversing the length
    of the row, and zeroing out the padding entries at the end.
    """
    _, T_y = Y.shape
    device = Y.device
    w = torch.tril(torch.ones((T_y, T_y), device=device))
    accumulated_matrix = (w @ mass_tensor(Y, device=device)
                          .transpose(-2, -1)).transpose(-2, -1)
    return accumulated_matrix.masked_fill(Y == 0, 0.0)


def create_fragments_tensor(Y: torch.Tensor) -> torch.Tensor:
    """
     Ions: b, y, b(+2), y(+2), b-H2O, y-H2O, b-NH3, y-NH3
     B * T_y * I * R(ange)
     B: Batch
     T_y: peptide length
     I: Ions count
     R: Range of weights: [w-1, w-0.5, w, w+0.5, w+1]
    """
    I = M.IONS_COUNT
    Range = M.RANGE_OF_WEIGHTS

    acc_tensor = create_accumulated_mass_tensor(Y)
    molecule_weights = acc_tensor.max(1).values
    frag_tensor = acc_tensor.unsqueeze(-1).expand(-1, -1, I).clone()

    # create and add y-ions to odd-columns
    mw_offset = torch.zeros_like(frag_tensor)
    for i in range(molecule_weights.size(0)):
        mw_offset[i, :, 1::2] = -molecule_weights[i]

    # b(+2) and y(+2)
    double_charges = torch.zeros_like(frag_tensor)
    double_charges[:, :, 2] += M.PROTON
    double_charges[:, :, 3] += M.PROTON
    # dividor
    double_charge_multiplier = torch.ones_like(frag_tensor)
    double_charge_multiplier[:, :, 2:4] = 1/2

    # add H2O and NH3
    water_ammuniom = torch.zeros_like(frag_tensor)
    water_ammuniom[:, :, 4:6] -= M.H2O
    water_ammuniom[:, :, 6:] -= M.NH3

    ss = torch.ones_like(mw_offset)
    ss[:, :, 1::2] = -1
    frags_b_y_ions = (frag_tensor + mw_offset) * ss
    frags_b_y_2ch = (frags_b_y_ions + double_charges) * double_charge_multiplier
    frags_b_y_2ch_w_a = frags_b_y_2ch + water_ammuniom

    # add Range dimension
    ranged = frags_b_y_2ch_w_a.unsqueeze(-1).expand(-1, -1, -1, Range).clone()
    ranged[:, :, :, 0] -= 1
    ranged[:, :, :, 1] -= 0.5
    ranged[:, :, :, 3] += 0.5
    ranged[:, :, :, 4] += 1

    zero_pad = ranged.masked_fill(frag_tensor.unsqueeze(-1).expand(-1, -1, -1, Range) == 0, 0.0) 
    return zero_pad.clamp(0, 10000)
