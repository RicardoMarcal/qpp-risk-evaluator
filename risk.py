import torch
import math

DEFAULT_EPS = 1e-10

def zRisk(mat, alpha, device, requires_grad=False, i=0):
    alpha_tensor = torch.tensor(
        [alpha], requires_grad=requires_grad, dtype=torch.float, device=device
    )
    si = torch.sum(mat[:, i])

    tj = torch.sum(mat, dim=1)
    n = torch.sum(tj)

    xij_eij = mat[:, i] - si * (tj / n)
    subden = si * (tj / n)
    den = torch.sqrt(subden + 1e-10)
    u = (den == 0) * torch.tensor(
        [9e10], dtype=torch.float, requires_grad=requires_grad, device=device
    )

    den = u + den
    div = xij_eij / den

    less0 = (mat[:, i] - si * (tj / n)) / (den) < 0
    less0 = alpha_tensor * less0

    z_risk = div * less0 + div
    z_risk = torch.sum(z_risk)

    return z_risk


def geoRisk(mat, alpha, device, requires_grad=False, i=0, do_root=False):
    mat = mat * (mat > 0)
    si = torch.sum(mat[:, i])
    z_risk = zRisk(mat, alpha, device, requires_grad=requires_grad, i=i)

    num_queries = mat.shape[0]
    value = z_risk / num_queries
    m = torch.distributions.normal.Normal(
        torch.tensor([0.0], device=device), torch.tensor([1.0], device=device)
    )
    ncd = m.cdf(value)
    
    if do_root:
        return torch.sqrt((si / num_queries) * ncd + DEFAULT_EPS)
    return (si / num_queries) * ncd

def URisk(apList, baseRiskList, alpha):
    size = len(apList)
    urisk=0
    for i in range(0, size):
        delta = apList[i] - baseRiskList[i]
        if (apList[i] < baseRiskList[i]):
            delta = (1+alpha)*delta
        urisk=urisk+delta

    urisk=urisk/size

    return urisk

def TRisk(apList, baseRiskList, alpha):
    size = len(apList)
    deltaList = []
    urisk=0
    trisk=0
    for i in range(0, size):
        delta = apList[i] - baseRiskList[i]
        if (apList[i] < baseRiskList[i]):
            delta = (1+alpha)*delta
        deltaList.append(delta)
        urisk=urisk+delta

    urisk=urisk/size
    for i in range(0, size):
        dif = deltaList[i]-urisk
        trisk=trisk + (dif * dif)

    trisk=math.sqrt(trisk/size)
    trisk=(math.sqrt(size)/trisk)*urisk
    return trisk