from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Dict, List

from .types import AttendanceEntry, PersonRecord


class AttendanceManager:
    def __init__(self, min_confirm_frames: int = 5) -> None:
        self.min_confirm_frames = max(min_confirm_frames, 1)
        self._student_seen_counter = defaultdict(int)
        self._teacher_seen_counter = defaultdict(int)
        self._student_attendance: Dict[str, AttendanceEntry] = {}
        self._teacher_attendance: Dict[str, AttendanceEntry] = {}

    def _confidence_from_counter(self, count: int) -> float:
        # Confidence rises as repeated sightings exceed the minimum confirmation threshold.
        score = float(count) / float(self.min_confirm_frames * 2)
        return max(0.0, min(1.0, score))

    @staticmethod
    def _confidence_band(score: float) -> str:
        if score >= 0.75:
            return "high"
        if score >= 0.4:
            return "medium"
        return "low"

    def _mark(
        self,
        person: PersonRecord,
        seen_at: datetime,
        counter: defaultdict,
        bucket: Dict[str, AttendanceEntry],
    ) -> None:
        counter[person.person_id] += 1
        confidence = self._confidence_from_counter(counter[person.person_id])
        confidence_band = self._confidence_band(confidence)
        if person.person_id not in bucket and counter[person.person_id] >= self.min_confirm_frames:
            bucket[person.person_id] = AttendanceEntry(
                person_id=person.person_id,
                name=person.name,
                role=person.role,
                status="present",
                first_seen=seen_at,
                last_seen=seen_at,
                confidence=confidence,
                confidence_band=confidence_band,
            )
        elif person.person_id in bucket:
            bucket[person.person_id].last_seen = seen_at
            bucket[person.person_id].confidence = confidence
            bucket[person.person_id].confidence_band = confidence_band

    def mark_seen(self, person: PersonRecord, seen_at: datetime) -> None:
        if person.role == "student":
            self._mark(person, seen_at, self._student_seen_counter, self._student_attendance)
        elif person.role == "teacher":
            self._mark(person, seen_at, self._teacher_seen_counter, self._teacher_attendance)

    def get_present_students(self) -> List[AttendanceEntry]:
        return sorted(self._student_attendance.values(), key=lambda x: x.person_id)

    def get_present_teachers(self) -> List[AttendanceEntry]:
        return sorted(self._teacher_attendance.values(), key=lambda x: x.person_id)

    def finalize_students(self, known_students: List[PersonRecord], include_absent: bool = True) -> List[AttendanceEntry]:
        results = {entry.person_id: entry for entry in self.get_present_students()}
        if include_absent:
            for person in known_students:
                if person.person_id not in results:
                    results[person.person_id] = AttendanceEntry(
                        person_id=person.person_id,
                        name=person.name,
                        role=person.role,
                        status="absent",
                        confidence=0.0,
                        confidence_band="low",
                    )
        return sorted(results.values(), key=lambda x: x.person_id)

    def finalize_teachers(self, known_teachers: List[PersonRecord], include_absent: bool = True) -> List[AttendanceEntry]:
        results = {entry.person_id: entry for entry in self.get_present_teachers()}
        if include_absent:
            for person in known_teachers:
                if person.person_id not in results:
                    results[person.person_id] = AttendanceEntry(
                        person_id=person.person_id,
                        name=person.name,
                        role=person.role,
                        status="absent",
                        confidence=0.0,
                        confidence_band="low",
                    )
        return sorted(results.values(), key=lambda x: x.person_id)