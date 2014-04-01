#pragma once

namespace loss_functions {

class square_loss {
public:
  inline double
  loss(double y, double haty) const
  {
    const double diff = (y - haty);
    return 0.5 * diff * diff;
  }

  inline double
  dloss(double y, double haty) const
  {
    const double diff = (y - haty);
    return -diff;
  }
};

class hinge_loss {
public:
  inline double
  loss(double y, double haty) const
  {
    const double z = y * haty;
    if (z > 1.0)
      return 0.0;
    return 1.0 - z;
  }

  inline double
  dloss(double y, double haty) const
  {
    const double z = y * haty;
    if (z > 1.0)
      return 0.0;
    return -y;
  }
};

class ramp_loss {
public:

  inline double
  loss(double y, double haty) const
  {
    const double z = y * haty;
    if (z > 1.0)
      return 0.0;
    if (z < -1.0)
      return 2.0;
    return 1. - z;
  }

  inline double
  dloss(double y, double haty) const
  {
    const double z = y * haty;
    if (z > 1.0 || z < -1.0)
      return 0.0;
    return -y;
  }
};

} // namespace loss_functions
