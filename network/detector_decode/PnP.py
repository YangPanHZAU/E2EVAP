import torch
def pnp(vertices, width, height):
    device = vertices.device
    batch_size = vertices.size(0)
    polygon_dimension = vertices.size(1)

    y_index = torch.arange(0, height).to(device)
    x_index = torch.arange(0, width).to(device)
    
    grid_y, grid_x = torch.meshgrid(y_index, x_index)
    xp = grid_x.unsqueeze(0).repeat(batch_size, 1, 1).float()
    yp = grid_y.unsqueeze(0).repeat(batch_size, 1, 1).float()

    result = torch.zeros((batch_size, height, width)).bool().to(device)

    j = polygon_dimension - 1
    for vn in range(polygon_dimension):
        from_x = vertices[:, vn, 0].unsqueeze(-1).unsqueeze(-1).repeat(1, height, width)        
        from_y = vertices[:, vn, 1].unsqueeze(-1).unsqueeze(-1).repeat(1, height, width)

        to_x = vertices[:, j, 0].unsqueeze(-1).unsqueeze(-1).repeat(1, height, width)
        to_y = vertices[:, j, 1].unsqueeze(-1).unsqueeze(-1).repeat(1, height, width)

        has_condition = torch.logical_and((from_y > yp) != (to_y > yp), xp < ((to_x - from_x) * (yp - from_y) / (to_y - from_y) + from_x))
        
        if has_condition.any():
            result[has_condition] = ~result[has_condition]

        j = vn

    signed_result = torch.zeros((batch_size, height, width), device=device)
    signed_result[result] = 1.0

    return signed_result.float()