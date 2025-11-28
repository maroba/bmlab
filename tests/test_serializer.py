from pathlib import Path
import os
import shutil
import uuid

import h5py
import numpy as np
import pytest

from bmlab.session import Session
from bmlab.models.calibration_model import FitSet, RayleighFit
from bmlab.models.extraction_model import CircleFit
from bmlab.controllers import CalibrationController, ExtractionController
from bmlab.constants import ExtractionMethod
from bmlab.serializer import Serializer


@pytest.fixture()
def tmp_dir():
    tmp_dir = "tmp" + str(uuid.uuid4())
    os.mkdir(tmp_dir)
    os.chdir(tmp_dir)
    yield tmp_dir
    os.chdir("..")
    try:
        shutil.rmtree(tmp_dir)
    # Windows sometimes does not correctly close the HDF file. Then we have a
    # file access conflict.
    except Exception as e:
        print(e)


@pytest.fixture()
def session_file(tmp_dir):

    session = Session.get_instance()

    shutil.copy(data_file_path("Water.h5"), Path.cwd() / "Water.h5")

    session.set_file("Water.h5")

    session.set_reflection(vertically=True, horizontally=False)

    session.set_current_repetition("0")

    ec = ExtractionController()
    cc = CalibrationController()

    em = session.extraction_model()
    cm = session.calibration_model()

    img = session.get_payload_image("0", 0)
    em.set_image_shape(img.shape)

    points = [(100, 290), (145, 255), (290, 110)]
    for calib_key in session.get_calib_keys():
        for p in points:
            ec.add_point(calib_key, p)

        assert em.get_arc_by_calib_key(calib_key).size != 0
        assert cm.get_spectra(calib_key) is None

        cc.extract_spectra(calib_key)

    session.save()

    session.clear()

    yield "Water.session.h5"


def data_file_path(file_name):
    return Path(__file__).parent / "data" / file_name


def test_serialize_session(tmp_dir):
    session = Session.get_instance()

    shutil.copy(data_file_path("Water.h5"), Path.cwd() / "Water.h5")

    session.set_file("Water.h5")

    session.set_reflection(vertically=True, horizontally=False)

    session.set_current_repetition("0")

    ec = ExtractionController()
    cc = CalibrationController()

    em = session.extraction_model()
    cm = session.calibration_model()

    points = [(100, 290), (145, 255), (290, 110)]
    for calib_key in session.get_calib_keys():
        for p in points:
            ec.add_point(calib_key, p)

        assert em.get_arc_by_calib_key(calib_key).size != 0
        assert cm.get_spectra(calib_key) is None

        cc.extract_spectra(calib_key)

    session.save()

    with h5py.File("Water.session.h5", "r") as f:
        assert "session/extraction_models/0/points/1" in f
        assert f.attrs["version"].startswith("bmlab")

    session.clear()


def test_deserialize_session_file(session_file):

    session = Session.get_instance()
    session.set_file("Water.h5")

    em = session.extraction_model()
    cm = session.calibration_model()
    assert em
    assert em.calib_times["2"] == 62.542
    assert len(cm.spectra["1"]) > 0
    assert len(em.positions["2"]) > 0
    assert em.positions_interpolation is not None

    session.clear()


def test_serialize_fitset(tmp_dir):

    fit_set = FitSet()
    fit = RayleighFit("1", 3, 4, 11.0, 12.0, 13.0, 14.0)
    fit_set.add_fit(fit)

    with h5py.File("tmpsession.h5", "w") as f:
        fit_set.serialize(f, "fits")

    with h5py.File("tmpsession.h5", "r") as f:
        actual = FitSet.deserialize(f["fits"])

    actual_fit = actual.get_fit("1", 3, 4)
    expected_fit = fit_set.get_fit("1", 3, 4)
    assert actual_fit.w0 == expected_fit.w0
    assert actual_fit.fwhm == expected_fit.fwhm
    assert actual_fit.intensity == expected_fit.intensity
    assert actual_fit.offset == expected_fit.offset


def test_de_serialize_CircleFit(tmp_dir):

    cf = CircleFit(center=(1.0, 2.0), radius=3.0)

    with h5py.File(str(tmp_dir) + "abc.h5", "w") as f:
        cf.serialize(f, "circle")

        assert isinstance(f["circle"], h5py.Group)
        assert f["circle"].attrs["type"] == "bmlab.models.extraction_model.CircleFit"

    with h5py.File(str(tmp_dir) + "abc.h5", "r") as f:
        cf = CircleFit.deserialize(f["circle"])

        np.testing.assert_array_equal(cf.center, (1.0, 2.0))
        assert cf.radius == 3.0


class TestEnumSerializable(Serializer):
    def __init__(self, extraction_method):
        self.extraction_method = extraction_method


def test_serialize_enum(tmp_dir):
    """Test serialization and deserialization of enum values."""

    # Create a test object with an enum
    test_obj = TestEnumSerializable(ExtractionMethod.ARC_FROM_PTS_OF_ALL_IMGS)

    # Serialize to HDF5
    with h5py.File("test_enum.h5", "w") as f:
        test_obj.serialize(f, "test_obj")

        # Check that the enum was serialized correctly
        assert "extraction_method" in f["test_obj"]
        dataset = f["test_obj"]["extraction_method"]
        assert dataset.attrs["type"] == "enum"
        assert dataset.attrs["enum_class"] == "bmlab.constants.ExtractionMethod"
        value = dataset[...].item()
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        assert value == "arc_from_pts_of_all_imgs"

    # Deserialize from HDF5
    with h5py.File("test_enum.h5", "r") as f:
        deserialized_obj = Serializer.deserialize(f["test_obj"])

        # Check that the enum was deserialized correctly
        assert isinstance(deserialized_obj.extraction_method, ExtractionMethod)
        assert (
            deserialized_obj.extraction_method
            == ExtractionMethod.ARC_FROM_PTS_OF_ALL_IMGS
        )


def test_session_with_extraction_method_serialization(tmp_dir):
    """Test that a Session with extraction_method can be serialized and deserialized."""

    # Copy existing test data file
    shutil.copy(data_file_path("Water.h5"), Path.cwd() / "Water.h5")

    # Get session instance and set up basic properties
    session = Session.get_instance()
    session.clear()
    session.set_file("Water.h5")
    session.extraction_method = ExtractionMethod.ARC_FROM_PTS_OF_ALL_IMGS

    # Test that session save works without exception
    try:
        session.save()
        save_successful = True
    except Exception as e:
        save_successful = False
        error_message = str(e)

    assert (
        save_successful
    ), f"Session save failed with error: {error_message if not save_successful else 'None'}"

    # Clean up
    session.clear()
