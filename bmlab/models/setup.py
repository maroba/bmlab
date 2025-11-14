import numpy as np

from math import cos
from scipy import interpolate

from bmlab.constants import ExtractionMethod
import bmlab.constants as constants
from bmlab.serializer import Serializer


class Setup(Serializer):

    def __init__(
        self,
        key,
        name,
        pixel_size,
        focal_length,
        vipa,
        calibration,
        temperature,
        extraction_method,
    ):
        """

        Parameters
        ----------
        key: str
            ID for the setup
        name: str
            Name of setup
        pixel_size: float
            pixel size of the camera [m]
        lambda0: float
            laser wavelength [m]
        focal_length: float
            focal length of the lens behind the VIPA [m]
        vipa
        calibration
        """
        self.key = key
        self.name = name
        self.pixel_size = pixel_size
        self.lambda0 = vipa.lambda0
        self.f0 = constants.c / self.lambda0
        self.focal_length = focal_length
        self.vipa = vipa
        self.calibration = calibration
        self.VIPA_PARAMS = self.init_vipa_params()
        # Default calibration temperature [°C]
        self.temperature = temperature
        self.extraction_method = extraction_method

    def post_deserialize(self):
        # Migrations from 0.3.0 to 0.4.0
        if not hasattr(self, "temperature"):
            self.temperature = 295.15

    def init_vipa_params(self):
        p1 = (
            2 * np.pi * self.vipa.n * self.vipa.d * np.cos(self.vipa.theta)
        ) / constants.c
        p2 = (
            -(2 * np.pi * self.vipa.n * self.vipa.d * np.tan(self.vipa.theta))
            / (constants.c * self.focal_length)
            * np.sqrt(1 - (self.vipa.n * np.sin(self.vipa.theta)) ** 2)
        )
        p3 = (
            -np.pi
            / constants.c
            * self.vipa.d
            * np.cos(self.vipa.theta)
            / (self.focal_length**2)
        )

        return p1, p2, p3

    def set_temperature(self, temperature):
        # Convert temperature from [°C] to [K]
        temperature = temperature + 273.15
        self.temperature = temperature
        # taken from
        # https://www.engineeringtoolbox.com/sound-speed-water-d_598.html
        # temperature water [K]
        water_t = [
            273.15,
            278.15,
            283.15,
            293.15,
            303.15,
            313.15,
            323.15,
            333.15,
            343.15,
            353.15,
            363.15,
            373.15,
        ]
        # sound velocity water [m/s]
        water_vs = [
            1403,
            1427,
            1447,
            1481,
            1507,
            1526,
            1541,
            1552,
            1555,
            1555,
            1550,
            1543,
        ]
        # Refractive index water [1]
        water_n = 1.3298
        water_f = interpolate.interp1d(water_t, water_vs)

        # taken from https://pubs.acs.org/doi/pdf/10.1021/je00054a002
        # temperature methanol [K]
        methanol_t = [274.74, 283.17, 293.15, 303.15, 313.11, 323.05, 332.95]
        # sound velocity methanol [m/s]
        methanol_vs = [1183.4, 1154.1, 1121.0, 1087.1, 1054.6, 1022.3, 990.3]
        # Refractive index methanol [1]
        methanol_n = 1.3234
        methanol_f = interpolate.interp1d(methanol_t, methanol_vs)

        water_shift = self.brillouin_shift(water_f(temperature), water_n)
        methanol_shift = self.brillouin_shift(methanol_f(temperature), methanol_n)

        self.calibration.set_shift_methanol(methanol_shift)
        self.calibration.set_shift_water(water_shift)
        self.calibration.update_calibration()

    def brillouin_shift(self, v, n):
        return 2 * cos(self.vipa.theta / 2) * n * v / self.vipa.lambda0


class EomSetup(Serializer):

    def __init__(self, key, name, poly_coefs):
        self.key = key
        self.name = name
        self.poly_coefs = poly_coefs


class VIPA(Serializer):

    def __init__(self, d, n, theta, order, lambda0):
        """Start values for VIPA fit

        Parameters
        ----------
        d : float
            width of the cavity [m]
        n : float
            refractive index of the cavity [-]
        theta : float
            angle [rad]
        order: int
            observed order of the VIPA spectrum
        """
        self.d = d
        self.n = n
        self.theta = theta
        self.order = order
        self.lambda0 = lambda0
        self.FSR = constants.c / (2 * self.n * self.d * np.cos(self.theta))
        self.m = round(constants.c / (self.lambda0 * self.FSR))


class Calibration(Serializer):

    def __init__(
        self,
        num_brillouin_samples,
        shift_methanol=None,
        shift_water=None,
        expected_shifts_polynomial=None,
        expected_shifts_values=None,
    ):
        """

        Parameters
        ----------
        num_brillouin_samples: int
            Number of samples
        shift_methanol: float
            ??
        shift_water: float
            ??
        """

        self.num_brillouin_samples = num_brillouin_samples
        self.shift_methanol = shift_methanol
        self.shift_water = shift_water
        self.expected_shifts_polynomial = None
        self.expected_shifts_interpolator = None

        if expected_shifts_polynomial is not None:
            self.expected_shifts_polynomial = np.poly1d(
                np.array(expected_shifts_polynomial[::-1])
            )
        elif expected_shifts_values is not None:
            # Create linear interpolator from (voltage, frequency) tuples
            voltages = np.array([v for v, f in expected_shifts_values])
            frequencies = np.array([f for v, f in expected_shifts_values])
            self.expected_shifts_interpolator = interpolate.interp1d(
                voltages, frequencies, kind="linear", fill_value="extrapolate"
            )
        else:
            self.expected_shifts_polynomial = None

        self._shifts = np.array([])
        self.orders = np.array([])

        self.update_calibration()

    def set_shift_water(self, shift_water):
        if self.shift_water is not None:
            self.shift_water = shift_water
        self.update_calibration()

    def set_shift_methanol(self, shift_methanol):
        if self.shift_methanol is not None:
            self.shift_methanol = shift_methanol
        self.update_calibration()

    @property
    def shifts(self):
        if self.expected_shifts_polynomial:
            return self.expected_shifts_polynomial
        elif self.expected_shifts_interpolator is not None:
            return self.expected_shifts_interpolator
        return self._shifts

    def update_calibration(self):
        # Construct array with the frequency shifts

        self._shifts = np.array([])
        if self.shift_methanol or self.shift_water:
            tmp = [self.shift_methanol, self.shift_water]

            self._shifts = np.full(2 + 2 * self.num_brillouin_samples, 0.0)
            for i in range(self.num_brillouin_samples):
                self._shifts[i + 1] = tmp[i]
                self._shifts[-1 * (i + 2)] = -1 * tmp[i]

        # The interference orders to which the peaks belong
        self.orders = np.full(2 + 2 * self.num_brillouin_samples, 0)
        self.orders[-(1 + self.num_brillouin_samples) :] = 1


AVAILABLE_SETUPS = [
    Setup(
        key="S0",
        name="780 nm @ Biotec R340",
        pixel_size=6.5e-6,
        focal_length=0.2,
        vipa=VIPA(
            d=0.006743,
            n=1.45367,
            theta=0.8 * 2 * np.pi / 360,
            order=0,
            lambda0=780.24e-9,
        ),
        calibration=Calibration(
            num_brillouin_samples=2, shift_methanol=3.78e9, shift_water=5.066e9
        ),
        temperature=295.15,
        extraction_method=ExtractionMethod.ARC_FROM_PTS_OF_AVG_IMG,
    ),
    Setup(
        key="S1",
        name="780 nm @ Biotec R340 old",
        pixel_size=6.5e-6,
        focal_length=0.2,
        vipa=VIPA(
            d=0.006743,
            n=1.45367,
            theta=0.8 * 2 * np.pi / 360,
            order=0,
            lambda0=780.24e-9,
        ),
        calibration=Calibration(num_brillouin_samples=1, shift_methanol=3.78e9),
        temperature=295.15,
        extraction_method=ExtractionMethod.ARC_FROM_PTS_OF_AVG_IMG,
    ),
    Setup(
        key="S2",
        name="532 nm @ Biotec R314",
        pixel_size=6.5e-6,
        focal_length=0.2,
        vipa=VIPA(
            d=0.003371, n=1.46071, theta=0.8 * 2 * np.pi / 360, order=0, lambda0=532e-9
        ),
        calibration=Calibration(
            num_brillouin_samples=2, shift_methanol=5.54e9, shift_water=7.43e9
        ),
        temperature=295.15,
        extraction_method=ExtractionMethod.ARC_FROM_PTS_OF_AVG_IMG,
    ),
    Setup(
        key="S3",
        name="MPZPM FOB",
        pixel_size=6.5e-6,
        focal_length=0.2,
        vipa=VIPA(
            d=0.006743,
            n=1.45367,
            theta=0.8 * 2 * np.pi / 360,
            order=0,
            lambda0=780.24e-9,
        ),
        calibration=Calibration(
            num_brillouin_samples=1,
            # coefficients for polynomial volt => frequency shift [Hz], lowest order first
            expected_shifts_polynomial=[
                3.40049,
                0.596645,
                -0.106149,
                0.0228064,
                -0.00273963,
                0.000168553,
                -4.19117e-6,
            ],
        ),
        temperature=295.15,
        extraction_method=ExtractionMethod.ARC_FROM_PTS_OF_ALL_IMGS,
    ),
    Setup(
        key="S4",
        name="MPZPM FOB, Fit 1",
        pixel_size=6.5e-6,
        focal_length=0.2,
        vipa=VIPA(
            d=0.006743,
            n=1.45367,
            theta=0.8 * 2 * np.pi / 360,
            order=0,
            lambda0=780.24e-9,
        ),
        calibration=Calibration(
            num_brillouin_samples=1,
            # coefficients for polynomial volt => frequency shift [Hz], lowest order first
            expected_shifts_polynomial=[
                3.78509,
                0.268194,
                0.0387374,
                -0.00875844,
                0.000894938,
                -0.0000432597,
                7.69557e-7,
            ],
        ),
        temperature=295.15,
        extraction_method=ExtractionMethod.ARC_FROM_PTS_OF_ALL_IMGS,
    ),
    Setup(
        key="S5",
        name="MPZPM FOB, Fit 2",
        pixel_size=6.5e-6,
        focal_length=0.2,
        vipa=VIPA(
            d=0.006743,
            n=1.45367,
            theta=0.8 * 2 * np.pi / 360,
            order=0,
            lambda0=780.24e-9,
        ),
        calibration=Calibration(
            num_brillouin_samples=1,
            # coefficients for polynomial volt => frequency shift [Hz], lowest order first
            expected_shifts_polynomial=[
                3.66825,
                0.268194,
                0.0387374,
                -0.00875844,
                0.000894938,
                -0.0000432597,
                7.69557e-7,
            ],
        ),
        temperature=295.15,
        extraction_method=ExtractionMethod.ARC_FROM_PTS_OF_ALL_IMGS,
    ),
    Setup(
        key="S6",
        name="MPZPM FOB, No Fit + Extrapolation",
        pixel_size=6.5e-6,
        focal_length=0.2,
        vipa=VIPA(
            d=0.006743,
            n=1.45367,
            theta=0.8 * 2 * np.pi / 360,
            order=0,
            lambda0=780.24e-9,
        ),
        calibration=Calibration(
            num_brillouin_samples=1,
            # coefficients for polynomial volt => frequency shift [Hz], lowest order first
            expected_shifts_values=[
                (0, 3.669157366),
                (0.1, 3.702491629),
                (0.2, 3.735825893),
                (0.3, 3.769160156),
                (0.4, 3.80249442),
                (0.5, 3.835828683),
                (0.6, 3.869162946),
                (0.7, 3.90249721),
                (0.8, 3.935831473),
                (0.9, 3.969165737),
                (1, 4.0025),
                (1.1, 4.035834263),
                (1.2, 4.069168527),
                (1.3, 4.10250279),
                (1.4, 4.135837054),
                (1.5, 4.169171317),
                (1.6, 4.20250558),
                (1.7, 4.235839844),
                (1.8, 4.269174107),
                (1.9, 4.302508371),
                (2, 4.335842634),
                (2.1, 4.369176897),
                (2.2, 4.402511161),
                (2.3, 4.435845424),
                (2.4, 4.469179688),
                (2.5, 4.502513951),
                (2.6, 4.535848214),
                (2.7, 4.569182478),
                (2.8, 4.602516741),
                (2.9, 4.635851004),
                (3, 4.669185268),
                (3.1, 4.702519531),
                (3.2, 4.735853795),
                (3.3, 4.769188058),
                (3.4, 4.802522321),
                (3.5, 4.835856585),
                (3.6, 4.869190848),
                (3.7, 4.902525112),
                (3.8, 4.935859375),
                (3.9, 4.968984375),
                (4, 5.0021875),
                (4.1, 5.035),
                (4.2, 5.068125),
                (4.3, 5.101015625),
                (4.4, 5.133828125),
                (4.5, 5.1665625),
                (4.6, 5.199375),
                (4.7, 5.231875),
                (4.8, 5.26453125),
                (4.9, 5.29703125),
                (5, 5.329296875),
                (5.1, 5.361640625),
                (5.2, 5.3940625),
                (5.3, 5.42625),
                (5.4, 5.458203125),
                (5.5, 5.490390625),
                (5.6, 5.5221875),
                (5.7, 5.5540625),
                (5.8, 5.585859375),
                (5.9, 5.6175),
                (6, 5.648984375),
                (6.1, 5.680546875),
                (6.2, 5.71203125),
                (6.3, 5.743125),
                (6.4, 5.774453125),
                (6.5, 5.805625),
                (6.6, 5.8365625),
                (6.7, 5.867421875),
                (6.8, 5.898359375),
                (6.9, 5.9290625),
                (7, 5.95953125),
                (7.1, 5.990078125),
                (7.2, 6.0203125),
                (7.3, 6.050546875),
                (7.4, 6.080703125),
                (7.5, 6.110859375),
                (7.6, 6.140625),
                (7.7, 6.170390625),
                (7.8, 6.19984375),
                (7.9, 6.229375),
                (8, 6.258828125),
                (8.1, 6.288359375),
                (8.2, 6.3178125),
                (8.3, 6.346953125),
                (8.4, 6.37609375),
                (8.5, 6.405078125),
                (8.6, 6.43390625),
                (8.7, 6.46265625),
                (8.8, 6.491328125),
                (8.9, 6.51984375),
                (9, 6.548203125),
                (9.1, 6.576484375),
                (9.2, 6.604609375),
                (9.3, 6.632734375),
                (9.4, 6.660703125),
                (9.5, 6.688828125),
                (9.6, 6.716640625),
                (9.7, 6.74421875),
                (9.8, 6.7715625),
                (9.9, 6.798828125),
                (10, 6.825859375),
            ],
        ),
        temperature=295.15,
        extraction_method=ExtractionMethod.ARC_FROM_PTS_OF_ALL_IMGS,
    ),
]
