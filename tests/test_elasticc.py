import pytest

from naptime.elasticc import CLASS_NAMES, NUM_ELASTICC_CLASSES, RELEASE_TO_CLASS


class TestClassNames:
    def test_is_list_of_strings(self):
        assert isinstance(CLASS_NAMES, list)
        assert all(isinstance(n, str) for n in CLASS_NAMES)

    def test_length_matches_num_classes(self):
        assert len(CLASS_NAMES) == NUM_ELASTICC_CLASSES

    def test_no_duplicates(self):
        assert len(CLASS_NAMES) == len(set(CLASS_NAMES))

    def test_non_empty(self):
        assert all(len(n) > 0 for n in CLASS_NAMES)


class TestReleaseToClass:
    def test_is_dict(self):
        assert isinstance(RELEASE_TO_CLASS, dict)

    def test_keys_are_strings(self):
        assert all(isinstance(k, str) for k in RELEASE_TO_CLASS)

    def test_values_are_valid_class_ids(self):
        for key, class_id in RELEASE_TO_CLASS.items():
            assert isinstance(class_id, int), f"{key} maps to non-int {class_id}"
            assert (
                0 <= class_id < NUM_ELASTICC_CLASSES
            ), f"{key} maps to out-of-range class_id {class_id}"

    def test_non_empty(self):
        assert len(RELEASE_TO_CLASS) > 0

    def test_all_classes_reachable(self):
        """Every class in CLASS_NAMES should be reachable from at least one release key."""
        reachable = set(RELEASE_TO_CLASS.values())
        for class_id in range(NUM_ELASTICC_CLASSES):
            assert class_id in reachable, (
                f"class_id {class_id} ({CLASS_NAMES[class_id]}) "
                f"has no RELEASE_TO_CLASS entry"
            )


class TestNumClasses:
    def test_positive_integer(self):
        assert isinstance(NUM_ELASTICC_CLASSES, int)
        assert NUM_ELASTICC_CLASSES > 0
