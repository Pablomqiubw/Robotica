"""
irb120_traj.py
Trayectoria LSPB (Linear Segment with Parabolic Blends) en espacio articular,
SINCRONIZADA: las 6 juntas arrancan y terminan al mismo tiempo T.

Idea: se disena un perfil escalar s(t) de 0 a 1 y se aplica a todas las juntas:
        q_i(t) = q0_i + s(t) * (qf_i - q0_i)
Asi cada junta mantiene su proporcion de movimiento (trayectoria recta en el
espacio articular) y todas llegan juntas.

Perfil de s(t) con tiempo total T y tiempo de mezcla (blend) tb. Sea D = T - tb:
    s''  = +A    para 0 <= t <  tb          (A = 1/(tb*D))
    s'   =  V    para tb <= t <= T - tb     (V = 1/D)   <- tramo lineal
    s''  = -A    para T - tb < t <= T
Condicion de factibilidad:  tb <= T/2  (equivale a tb <= D).

Limites (los entrega el usuario, NO se inventan aqui):
    |dq_i| * V <= vmax_i      ->  D >= vr = max_i |dq_i| / vmax_i
    |dq_i| * A <= amax_i      ->  tb*D >= ar = max_i |dq_i| / amax_i
Tiempo minimo: D = max(vr, sqrt(ar)),  tb = ar / D,  T = D + tb.
  - Si vr >= sqrt(ar): hay tramo lineal (perfil trapezoidal).
  - Si no: perfil triangular (tb = D, T = 2*tb).

Entradas : q0, qf (6,) rad ; vmax, amax (6,) rad/s y rad/s^2
Salidas  : at(t) -> q, qd, qdd (6,) ; sample(dt) -> t, Q, Qd, Qdd
"""
import numpy as np


class LSPB:
    def __init__(self, q0, qf, vmax, amax):
        self.q0 = np.asarray(q0, dtype=float)
        self.qf = np.asarray(qf, dtype=float)
        self.dq = self.qf - self.q0
        vmax = np.asarray(vmax, dtype=float)
        amax = np.asarray(amax, dtype=float)
        if np.any(vmax <= 0) or np.any(amax <= 0):
            raise ValueError("vmax y amax deben ser > 0 (datos del datasheet/URDF)")

        ad = np.abs(self.dq)
        vr = float(np.max(ad / vmax))
        ar = float(np.max(ad / amax))
        if vr < 1e-12:                      # no hay movimiento
            self.T = self.tb = self.D = 0.0
            return
        self.D = max(vr, np.sqrt(ar))
        self.tb = ar / self.D
        self.T = self.D + self.tb
        self.A = 1.0 / (self.tb * self.D)   # s'' en la mezcla
        self.V = 1.0 / self.D               # s' en el tramo lineal

    def _s(self, t):
        """Devuelve (s, s', s'') del perfil escalar en el instante t."""
        T, tb, A, V = self.T, self.tb, self.A, self.V
        t = float(np.clip(t, 0.0, T))
        if t < tb:                                   # aceleracion
            return 0.5 * A * t * t, A * t, A
        if t <= T - tb:                              # velocidad constante
            return 0.5 * A * tb * tb + V * (t - tb), V, 0.0
        tau = T - t                                  # desaceleracion
        return 1.0 - 0.5 * A * tau * tau, A * tau, -A

    def at(self, t):
        """Posicion, velocidad y aceleracion articulares en el instante t."""
        if self.T == 0.0:
            return self.q0.copy(), np.zeros(6), np.zeros(6)
        s, sd, sdd = self._s(t)
        return self.q0 + s * self.dq, sd * self.dq, sdd * self.dq

    def sample(self, dt):
        """Muestrea toda la trayectoria. Devuelve t (N,), Q, Qd, Qdd (N,6)."""
        n = max(int(np.ceil(self.T / dt)), 1)
        t = np.linspace(0.0, self.T, n + 1)
        out = [self.at(ti) for ti in t]
        Q, Qd, Qdd = (np.array([o[k] for o in out]) for k in range(3))
        return t, Q, Qd, Qdd
