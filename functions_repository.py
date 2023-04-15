def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(math.radians(-angle)) * (px - ox) - math.sin(math.radians(-angle)) * (py - oy)
    qy = oy + math.sin(math.radians(-angle)) * (px - ox) + math.cos(math.radians(-angle)) * (py - oy)
    return qx, qy


def cart_2_polar(cart_x, cart_y):
    r = math.sqrt(cart_x ** 2 + cart_y ** 2)
    teta = 180 + math.atan(cart_y / cart_x)
    return r, teta


def polar_2_cart(polar_r, polar_teta):
    x = polar_r * np.cos(polar_teta)
    y = polar_r * np.sin(polar_teta)
    return x, y