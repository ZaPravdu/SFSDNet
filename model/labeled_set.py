class LabeledSet:
    """Tracks labeled sample IDs per scene using a ratio.

    Accepts a ratio and per-scene total counts, then internally computes
    per-scene max budgets: max(1, floor(ratio * total)).
    """
    def __init__(self, ratio=0, scene_totals=None):
        self._ids = set()
        self._counts = {}
        self.ratio = ratio
        self._per_scene_max = {}
        if scene_totals and ratio > 0:
            for scene, total in scene_totals.items():
                self._per_scene_max[scene] = max(1, int(ratio * total))

    def add(self, sid):
        scene = sid.split('/')[0]
        max_count = self._per_scene_max.get(scene, 0)
        if self._counts.get(scene, 0) < max_count:
            self._ids.add(sid)
            self._counts[scene] = self._counts.get(scene, 0) + 1

    def __contains__(self, sid):
        return sid in self._ids

    def check_consistency(self):
        """Assert _ids per scene match budgets and _counts.

        Raises AssertionError on any mismatch — fail-fast.
        """
        per_scene = {}
        for sid in self._ids:
            scene = sid.split('/')[0]
            per_scene[scene] = per_scene.get(scene, 0) + 1

        # _ids per scene must match _counts
        for scene, expected in self._counts.items():
            actual = per_scene.get(scene, 0)
            assert actual == expected, (
                f"Scene '{scene}': _ids has {actual} entries, "
                f"but _counts says {expected}"
            )

        # No scene may exceed its budget
        for scene, actual in per_scene.items():
            budget = self._per_scene_max.get(scene, 0)
            assert actual <= budget, (
                f"Scene '{scene}': {actual} labeled samples "
                f"exceeds budget {budget}"
            )

        # Verify _per_scene_max was computed correctly from ratio
        for scene, budget in self._per_scene_max.items():
            assert budget >= 1, (
                f"Scene '{scene}' budget is {budget}, expected at least 1"
            )

    def serialize(self):
        return {'ids': list(self._ids), 'counts': self._counts,
                'ratio': self.ratio,
                '_per_scene_max': self._per_scene_max}

    def deserialize(self, data):
        self._ids = set(data['ids'])
        self._counts = data['counts']
        if 'ratio' in data:
            self.ratio = data['ratio']
        if '_per_scene_max' in data:
            self._per_scene_max = data['_per_scene_max']
