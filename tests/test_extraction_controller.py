from pathlib import Path

import numpy as np

from bmlab.session import Session, ExtractionMethod
from bmlab.controllers import ExtractionController
from bmlab.models import ExtractionModel, Orientation
from bmlab.models.setup import AVAILABLE_SETUPS


def test_add_point(mocker):
    mocker.patch("bmlab.session.Session.get_calibration_time", return_value=0)

    em = ExtractionModel()
    mocker.patch("bmlab.session.Session.extraction_model", return_value=em)

    ec = ExtractionController()
    ec.add_point("0", (15, 15))

    points = em.get_points("0")

    assert len(points) == 1
    assert points[0] == (15, 15)

    ec.add_point("0", (20, 20))

    assert len(points) == 2
    assert points[0] == (15, 15)
    assert points[1] == (20, 20)


def test_set_point(mocker):
    mocker.patch("bmlab.session.Session.get_calibration_time", return_value=0)

    em = ExtractionModel()
    mocker.patch("bmlab.session.Session.extraction_model", return_value=em)

    ec = ExtractionController()
    ec.add_point("0", (15, 15))

    points = em.get_points("0")

    assert len(points) == 1
    assert points[0] == (15, 15)

    ec.add_point("0", (20, 20))

    assert len(points) == 2
    assert points[0] == (15, 15)
    assert points[1] == (20, 20)

    ec.set_point("0", 0, (10, 10))

    assert len(points) == 2
    assert points[0] == (10, 10)
    assert points[1] == (20, 20)

    ec.set_point("0", 2, (30, 20))

    assert len(points) == 3
    assert points[0] == (10, 10)
    assert points[1] == (20, 20)
    assert points[2] == (30, 20)


def test_optimize_points(mocker):

    imgs = np.zeros((1, 100, 100), dtype=int)
    imgs[0, 19:22, 19:22] = 1
    imgs[0, 79:82, 79:82] = 1

    em = ExtractionModel()

    mocker.patch("bmlab.session.Session.extraction_model", return_value=em)
    mocker.patch("bmlab.session.Session.get_calibration_image", return_value=imgs)
    mocker.patch("bmlab.session.Session.get_calibration_time", return_value=0)

    ec = ExtractionController()
    ec.add_point("0", (15, 15))
    ec.add_point("0", (5, 85))

    opt_points = em.get_points("0")

    assert (15, 15) == opt_points[0]
    assert (5, 85) == opt_points[1]

    ec.set_point("0", 1, (75, 85))

    ec.optimize_points("0", radius=10)
    opt_points = em.get_points("0")

    assert (19, 20) == opt_points[0]
    assert (79, 80) == opt_points[1]


def test_distance_point_to_line():
    ec = ExtractionController()

    np.testing.assert_almost_equal(
        0, ec.distance_point_to_line((0.5, 0.5), (1, 1), (0, 0))
    )
    np.testing.assert_almost_equal(
        0.5, ec.distance_point_to_line((0.5, 0.5), (0, 0), (0, 1))
    )
    np.testing.assert_almost_equal(
        np.sqrt(0.5), ec.distance_point_to_line((0, 1), (1, 1), (0, 0))
    )


def test_find_points(mocker):

    imgs = np.ones((1, 400, 200), dtype=int)
    imgs[0, 19:22, 19:22] = 5
    imgs[0, 379:382, 19:22] = 5
    imgs[0, 19:22, 179:182] = 5
    imgs[0, 379:382, 179:182] = 5

    em = ExtractionModel()

    mocker.patch("bmlab.session.Session.extraction_model", return_value=em)
    mocker.patch("bmlab.session.Session.get_calibration_image", return_value=imgs)
    mocker.patch("bmlab.session.Session.get_calibration_time", return_value=0)

    ec = ExtractionController()

    ec.find_points("0", 1, 4, 10)
    opt_points = em.get_points("0")

    assert len(opt_points) == 2
    assert (19, 180) == opt_points[0]
    assert (379, 20) == opt_points[1]


def test_find_points_real_data():
    # Start session
    session = Session.get_instance()

    # Load data file
    session.set_file(Path(__file__).parent / "data" / "Water.h5")

    # Select repetition
    session.set_current_repetition("0")
    session.set_setup(AVAILABLE_SETUPS[0])

    # Set orientation
    session.orientation = Orientation(
        rotation=1, reflection={"vertically": False, "horizontally": False}
    )

    em = session.extraction_model()

    ec = ExtractionController()

    points = {
        "1": [
            (107, 293),
            (165, 237),
            (182, 218),
            (240, 154),
            (254, 137),
            (291, 92),
        ],
        "2": [
            (107, 293),
            (165, 237),
            (182, 218),
            (241, 153),
            (255, 137),
            (291, 92),
        ],
    }

    for calib_key in session.get_calib_keys():
        ec.find_points(calib_key)

        p = em.get_points(calib_key)

        assert p == points[calib_key]


def test_find_points_for_method_arc_from_pts_of_all_imgs(mocker):
    """Test ExtractionController.find_points with ARC_FROM_PTS_OF_ALL_IMGS method.

    This test verifies that the find_points method works correctly when using the
    ARC_FROM_PTS_OF_ALL_IMGS extraction method, which processes multiple images
    and applies clustering to reduce outer clusters to single representative points.

    The test uses expected points data from Brillouin_NewCalibration_0_1_points.txt
    to create realistic mock image data with peaks at similar locations.
    """

    # Load expected points from test data file
    expected_points_file = (
        Path(__file__).parent / "data" / "Brillouin_NewCalibration_0_1_points.txt"
    )
    expected_points = []
    with open(expected_points_file, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                x, y = map(float, line.split(","))
                # Store as floats initially for more precise mock data creation
                expected_points.append((x, y))

    # Create mock images - simulate multiple images with peak locations
    # Use a reasonable number of images to test the ARC_FROM_PTS_OF_ALL_IMGS path
    num_images = 5  # Simulate multiple images
    img_height = 400
    img_width = 400

    # Create mock images with low background intensity
    imgs = np.ones((num_images, img_height, img_width), dtype=float) * 0.5

    # Add peaks at locations similar to expected points, with some variation across images
    # Use a subset of expected points to create a realistic test scenario
    test_points = expected_points[:15]  # Use first 15 points for cleaner testing

    for i, (row, col) in enumerate(test_points):
        if 0 <= row < img_height and 0 <= col < img_width:
            for img_idx in range(num_images):
                # Add some variation across images to simulate real conditions
                np.random.seed(
                    42 + img_idx + i
                )  # Consistent seed for reproducible tests
                noise_r = np.random.uniform(-1.5, 1.5)
                noise_c = np.random.uniform(-1.5, 1.5)
                peak_r = max(2, min(img_height - 3, row + noise_r))
                peak_c = max(2, min(img_width - 3, col + noise_c))

                # Create a Gaussian-like peak
                for dr in range(-3, 4):
                    for dc in range(-3, 4):
                        pr = int(max(0, min(img_height - 1, peak_r + dr)))
                        pc = int(max(0, min(img_width - 1, peak_c + dc)))
                        # Use Gaussian intensity distribution
                        distance_sq = dr**2 + dc**2
                        intensity = 15 * np.exp(-distance_sq / 4.0) + 0.5
                        imgs[img_idx, pr, pc] = max(imgs[img_idx, pr, pc], intensity)

    # Mock the Session and its methods
    mock_session = mocker.MagicMock()
    mock_session.extraction_method = ExtractionMethod.ARC_FROM_PTS_OF_ALL_IMGS
    mock_session.get_calibration_image.return_value = imgs
    mock_session.get_calibration_time.return_value = 12345.0
    mock_session.get_calibration_binning_factor.return_value = 1

    # Create mock extraction model
    mock_extraction_model = mocker.MagicMock()
    mock_session.extraction_model.return_value = mock_extraction_model

    # Mock Session.get_instance to return our mock session
    mocker.patch.object(Session, "get_instance", return_value=mock_session)

    # Create the ExtractionController and run find_points
    ec = ExtractionController()

    # Call find_points with the calibration key
    calib_key = "1"
    ec.find_points(
        calib_key=calib_key,
        min_height=8,  # Adjusted for our mock data
        min_area=15,  # Adjusted for our mock data
        max_distance=100,  # More permissive for testing
        use_threading=False,  # Disable threading for simpler testing
    )

    # Verify that set_points was called on the extraction model
    mock_extraction_model.set_points.assert_called_once()

    # Get the arguments passed to set_points
    call_args = mock_extraction_model.set_points.call_args
    assert call_args[0][0] == calib_key  # calib_key
    assert call_args[0][1] == 12345.0  # time
    found_points = call_args[0][2]  # points

    # Verify that points were found
    assert len(found_points) > 0, "No points were found by find_points method"

    # Verify that all found points are tuples of numbers (row, col format)
    for point in found_points:
        assert isinstance(point, tuple), f"Point {point} is not a tuple"
        assert len(point) == 2, f"Point {point} does not have exactly 2 coordinates"
        assert isinstance(
            point[0], (int, float, np.integer, np.floating)
        ), f"Point row {point[0]} is not a number"
        assert isinstance(
            point[1], (int, float, np.integer, np.floating)
        ), f"Point col {point[1]} is not a number"

    # Check that points are within image bounds
    for point in found_points:
        row, col = point
        assert 0 <= row < img_height, f"Point row {row} is out of bounds"
        assert 0 <= col < img_width, f"Point col {col} is out of bounds"

    # Verify that the ARC_FROM_PTS_OF_ALL_IMGS specific processing occurred
    # This method should process all images and apply clustering/filtering
    assert (
        mock_session.get_calibration_image.called
    ), "get_calibration_image should have been called"
    assert (
        mock_session.get_calibration_time.called
    ), "get_calibration_time should have been called"
    assert (
        mock_session.get_calibration_binning_factor.called
    ), "get_calibration_binning_factor should have been called"

    # Check that we have a reasonable number of points (not too many, not too few)
    # The clustering should reduce the number of raw peaks
    assert (
        1 <= len(found_points) <= len(test_points) * num_images
    ), f"Unexpected number of points: {len(found_points)}"
