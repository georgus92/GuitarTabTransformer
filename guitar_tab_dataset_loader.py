"""
Utility functions for parsing JAMS “note_tab” annotations and turning them
into a compact, token-based representation suitable for sequence models.

Main components
---------------
* GuitarTabTokenizer – bidirectional text ↔ integer vocabulary mapping
* parse_note_effect      – maps raw JAMS effect dicts to canonical strings
* parse_note_tab_from_jams – pulls note events from a .jams file and
                             converts them to (string, fret, Δt, effect) tuples
"""

import jams          # JAMS ⇢ standard JSON-based music annotation format
import json
import os

# Tell `jams` where to find the custom “note_tab” schema so it can validate
# annotations when we load them.
jams.schema.add_namespace(os.path.join('.', 'note_tab.json'))


class GuitarTabTokenizer:
    """
    Converts between human-readable tab tokens (e.g. “string_3_fret_5”) and
    integer IDs required by ML models.  Also handles timing “shift_x” tokens
    and many guitar-specific playing techniques.
    """
    def __init__(self, vocab_path: str):
        # Load the pre-built {token → id} mapping
        with open(vocab_path, "r") as f:
            self.token_to_id = json.load(f)

        # Create the reverse mapping on-the-fly
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

        # Build a fast-lookup set containing only technique / articulation tokens.
        # We ignore meta-symbols like PAD, BOS (begin-of-sequence), etc.
        self.effects = {
            t for t in self.token_to_id
            if t not in {"PAD", "EOS", "BOS", "UNK"} and (
                t.startswith("slide_") or
                t in [
                    "hammer", "slide_out_downwards", "slide_into_from_below",
                    "slide_shift", "sustain", "vibrato", "palm_mute",
                    "harmonic", "ghost_note", "staccato", "bend", "grace",
                    "trill", "tremolo_picking", "let_ring",
                    "heavy_accentuated", "accentuated"
                ]
            )
        }

    def encode(self, tab_sequence):
        """
        Convert a parsed tab event sequence into integer IDs.

        Parameters
        ----------
        tab_sequence : list of tuples
            Each tuple = (string, fret, shift_ms, effect)
              string     ∈ {1..6}  (1 = high-E, 6 = low-E)
              fret       ∈ ℕ (0 = open string)
              shift_ms   = silence / time gap before the note, in milliseconds
              effect     = token name, or None
        """
        tokens = ["BOS"]  # start-of-sequence

        for entry in tab_sequence:
            string, fret, shift_ms, effect = entry

            # --- Quantise time gap into repeated “shift_205” plus a remainder ---
            # Each “shift_205” covers 2050 ms; the remainder is binned to 10 ms
            full_shifts = shift_ms // 2050
            remainder   = shift_ms % 2050

            tokens.extend(["shift_205"] * full_shifts)
            tokens.append(f"shift_{remainder // 10}")

            # Actual note token
            note_token = f"string_{string}_fret_{fret}"

            # Optional playing technique before the note
            if effect in self.effects:
                tokens.append(effect)

            tokens.append(note_token)

        tokens.append("EOS")  # end-of-sequence
        # Map to integer IDs, defaulting to UNK for unseen tokens
        return [self.token_to_id.get(tok, self.token_to_id["UNK"]) for tok in tokens]

    def decode(self, token_ids):
        """
        Reconstruct tab events from an integer ID sequence.

        Returns
        -------
        events : list of tuples (string, fret, shift_ms, effect)
        """
        events   = []
        i        = 0
        shift_ms = 0  # running total of accumulated “shift_*” durations

        while i < len(token_ids):
            tok = self.id_to_token.get(token_ids[i], "UNK")

            # 1) Timing token: add to current offset and continue
            if tok.startswith("shift_"):
                shift_ms += int(tok.split("_")[1]) * 10  # 10 ms resolution
                i += 1
                continue

            # 2) (Possible) effect token – lookahead by one
            effect = None
            if tok in self.effects:
                effect = tok
                i += 1
                # Safeguard: if sequence ends with an effect token, stop
                if i < len(token_ids):
                    tok = self.id_to_token.get(token_ids[i], "UNK")
                else:
                    print("decode: sequence ended after effect token:", tok)
                    break

            # 3) Note token – extract string & fret numbers
            if tok.startswith("string_"):
                _, string, _, fret = tok.split("_")
                events.append((int(string), int(fret), shift_ms, effect))
                shift_ms = 0  # reset Δt after consuming it

            i += 1

        return events

def parse_note_effect(effect: dict) -> str:
    """
    Translate SynthTab’s nested effect descriptors into a single vocabulary
    token that matches those used by GuitarTabTokenizer.
    """
    # Slides have lots of sub-types
    if "slide" in effect:
        slide_type = effect["slide"]
        if "outDownwards"         in slide_type: return "slide_out_downwards"
        elif "intoFromBelow"      in slide_type: return "slide_into_from_below"
        elif "shiftSlideTo" in slide_type or "legatoSlideTo" in slide_type:
            return "slide_shift"

    if effect.get("palmMute")              : return "palm_mute"
    if effect.get("harmonic")              : return "harmonic"
    if effect.get("vibrato")               : return "vibrato"
    if effect.get("hammer")                : return "hammer"
    if "bend" in effect                    : return "bend"
    if effect.get("ghostNote")             : return "ghost_note"
    if effect.get("staccato")              : return "staccato"
    if effect.get("trill")                 : return "trill"
    if effect.get("tremoloPicking")        : return "tremolo_picking"
    if effect.get("grace")                 : return "grace"
    if effect.get("letRing")               : return "let_ring"
    if effect.get("accentuatedNote")       : return "accentuated"
    if effect.get("heavyAccentuatedNote")  : return "heavy_accentuated"
    # Default articulation when nothing else applies
    return "sustain"

def parse_note_tab_from_jams(jams_path: str):
    """
    Extract note-tab annotations from a JAMS file and compute per-note
    inter-onset intervals (shift_ms).

    Returns
    -------
    parsed_notes : list of tuples
        (string, fret, shift_ms, effect, cumulative_time_ms)
        * shift_ms        – gap since previous note (approx. /2 scaling, see below)
        * cumulative_time – running total (useful for debugging/plots)
    """
    jam = jams.load(jams_path)

    accurate_note_tab_annots = []

    # Iterate through every “note_tab” annotation track
    for annotation in jam.search(namespace="note_tab"):
        string = getattr(annotation.sandbox, "string_index", None)
        tuning = getattr(annotation.sandbox, "open_tuning", None)

        if string is None or tuning is None:
            continue

        # Sort note events by time for deterministic behaviour
        for note in sorted(annotation.data, key=lambda x: x.time):
            onset = int(note.time)              
            value = note.value                  # dict with fret & effects
            fret  = value.get("fret")
            effect = parse_note_effect(value)
            accurate_note_tab_annots.append((onset, string, fret, effect))

    # Global sort across all strings
    accurate_note_tab_annots.sort(key=lambda x: x[0])

    parsed_notes = []
    shift_ms     = 0
    last_time    = 0
    track_time   = 0  # cumulative

    for onset, string, fret, effect in accurate_note_tab_annots:
        ##############################################################
        # SynthTab annotations times didn't match audio times; 
        # dividing by 2 lines up events with audio better.
        ##############################################################
        shift_ms   = int((onset - last_time) / 2)
        track_time += shift_ms

        parsed_notes.append((
            string,                       
            fret,                         
            max(shift_ms, 0),             # guard against negatives
            effect,
            track_time                    # running total
        ))
        last_time = onset

    return parsed_notes
