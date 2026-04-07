from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from predictor_web.config import ALLOWED_DNA_CHARS


class FastaValidationError(ValueError):
    pass


@dataclass
class FastaRecord:
    sequence_id: int
    header: str
    sequence: str


def parse_and_validate_fasta(file_path: str, max_sequences: int) -> list[FastaRecord]:
    path = Path(file_path)
    if not path.exists() or path.stat().st_size == 0:
        raise FastaValidationError("Uploaded FASTA file is empty.")

    records: list[FastaRecord] = []
    current_header = None
    current_seq_parts: list[str] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue

            if line.startswith(">"):
                if current_header is not None:
                    records.append(_build_record(len(records) + 1, current_header, current_seq_parts, line_no))
                current_header = line[1:].strip()
                current_seq_parts = []
                if not current_header:
                    raise FastaValidationError(f"Header is empty near line {line_no}.")
            else:
                if current_header is None:
                    raise FastaValidationError("Invalid FASTA structure: sequence content appears before first header ('>').")
                current_seq_parts.append(line)

    if current_header is not None:
        records.append(_build_record(len(records) + 1, current_header, current_seq_parts, line_no=-1))

    if not records:
        raise FastaValidationError("No FASTA records found. Ensure each sequence starts with '>'.")
    if len(records) > max_sequences:
        raise FastaValidationError(
            f"Too many sequences: found {len(records)} records; maximum allowed is {max_sequences}."
        )

    return records


def _build_record(sequence_id: int, header: str, seq_parts: list[str], line_no: int) -> FastaRecord:
    sequence = "".join(seq_parts).strip()
    if not sequence:
        loc = f" near line {line_no}" if line_no > 0 else ""
        raise FastaValidationError(f"Sequence for header '{header}' is empty{loc}.")

    sequence = sequence.upper()
    invalid = sorted(set(sequence) - ALLOWED_DNA_CHARS)
    if invalid:
        allowed_str = "".join(sorted(ALLOWED_DNA_CHARS))
        raise FastaValidationError(
            f"Sequence '{header}' contains invalid DNA character(s): {', '.join(invalid)}. Allowed letters: {allowed_str}."
        )

    return FastaRecord(sequence_id=sequence_id, header=header, sequence=sequence)
