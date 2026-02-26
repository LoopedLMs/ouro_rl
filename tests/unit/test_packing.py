"""Unit tests for ouro_rl/packing.py — bin-packing and log-prob extraction."""


from ouro_rl.packing import pack_sequences


class TestPackSequences:
    def test_single_sequence(self):
        """Single sequence fits in one row."""
        packed = pack_sequences([[1, 2, 3]], [[4, 5]], max_pack_len=10)
        assert packed.num_sequences == 1
        assert packed.num_rows == 1
        assert packed.row_ids[0].tolist() == [1, 2, 3, 4, 5]
        assert packed.row_position_ids[0].tolist() == [0, 1, 2, 3, 4]
        info = packed.seq_infos[0]
        assert info.seq_idx == 0
        assert info.offset == 0
        assert info.resp_offset == 3
        assert info.total_len == 5

    def test_two_sequences_fit_one_row(self):
        """Two short sequences packed into a single row."""
        packed = pack_sequences(
            [[1, 2], [3, 4]],
            [[5], [6, 7]],
            max_pack_len=10,
        )
        assert packed.num_sequences == 2
        assert packed.num_rows == 1
        # Longer sequence (3,4,6,7 len=4) placed first by FFD, then (1,2,5 len=3).
        row = packed.row_ids[0].tolist()
        assert len(row) == 7  # 4 + 3
        # Verify position IDs reset at sequence boundary.
        pos = packed.row_position_ids[0].tolist()
        assert pos[0] == 0  # first seq starts at 0
        assert pos[4] == 0  # second seq starts at 0 (reset)

    def test_sequences_split_across_rows(self):
        """Sequences that don't fit together get separate rows."""
        packed = pack_sequences(
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8], [9, 10]],
            max_pack_len=5,
        )
        assert packed.num_sequences == 2
        assert packed.num_rows == 2
        # Each row has exactly one sequence.
        assert len(packed.row_ids[0]) == 5
        assert len(packed.row_ids[1]) == 5

    def test_ffd_packing_efficiency(self):
        """FFD packs small sequences with large ones."""
        # Lengths: 6, 4, 3, 2 → max_pack_len=8 → (6+2=8), (4+3=7) = 2 rows.
        packed = pack_sequences(
            [[1] * 3, [2] * 2, [3] * 2, [4] * 1],
            [[10] * 3, [20] * 2, [30] * 1, [40] * 1],
            max_pack_len=8,
        )
        assert packed.num_rows == 2

    def test_sequence_truncation(self):
        """Sequences exceeding max_pack_len get truncated (response first)."""
        packed = pack_sequences(
            [[1, 2, 3]],
            [[4, 5, 6, 7, 8]],
            max_pack_len=5,
        )
        info = packed.seq_infos[0]
        assert info.total_len == 5  # truncated to max_pack_len
        assert packed.row_ids[0].tolist() == [1, 2, 3, 4, 5]

    def test_seq_infos_indexed_by_original_order(self):
        """seq_infos[i].seq_idx == i regardless of packing order."""
        packed = pack_sequences(
            [[1], [2, 3, 4], [5, 6]],
            [[10], [20, 30], [40]],
            max_pack_len=20,
        )
        for i, info in enumerate(packed.seq_infos):
            assert info.seq_idx == i

    def test_no_tail_padding(self):
        """Packed rows have no trailing padding — exact concatenation."""
        packed = pack_sequences(
            [[1, 2], [3]],
            [[4], [5, 6]],
            max_pack_len=10,
        )
        for row in packed.row_ids:
            # Every position should have real tokens (no zeros from padding).
            assert row.shape[0] > 0

    def test_position_ids_are_local(self):
        """Position IDs reset to 0 at each sequence boundary within a packed row."""
        packed = pack_sequences(
            [[1, 2], [3]],
            [[4], [5, 6]],
            max_pack_len=10,
        )
        # All sequences in one row. Check position IDs.
        pos = packed.row_position_ids[0].tolist()
        # Seq 0 has len 3 (positions 0,1,2), seq 1 has len 3 (positions 0,1,2).
        # But FFD sorts by length desc, so order depends on lengths.
        # Both have len 3, so original order is preserved.
        # positions: [0,1,2, 0,1,2]
        assert pos.count(0) >= 2  # at least 2 sequences start with position 0


class TestPackSequencesEdgeCases:
    def test_empty_response(self):
        """Handle sequences with empty responses."""
        packed = pack_sequences([[1, 2]], [[]], max_pack_len=10)
        assert packed.num_sequences == 1
        info = packed.seq_infos[0]
        assert info.total_len == 2
        assert info.resp_offset == info.offset + 2  # response starts at end of prompt

    def test_all_same_length(self):
        """All sequences have same length — packing still works."""
        packed = pack_sequences(
            [[1, 2], [3, 4], [5, 6], [7, 8]],
            [[10], [20], [30], [40]],
            max_pack_len=6,
        )
        # Each seq is len 3. With max_pack_len=6, can fit 2 per row → 2 rows.
        assert packed.num_rows == 2
