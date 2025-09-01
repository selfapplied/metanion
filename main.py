import sys
import shutil
from sprixel import responsive, reflect_wave, dialog, sea, dusk, solar, squeeze, fuse
from sprixel2 import echo, mirror_of, dusk as dusk2, dict_ls, render, symmetry, stage, mark
from regex import validate_ce1_seed, parse_ce1_seed, extract_ce1_fields

# CE1 seed for self-verification
CE1_SEED = """CE1{
  lens=MAIN↔VERIFY | mode=SelfCheck | Ξ=main:integrity |
  data={m=0x1, α=0.000, E=100.0} |
  ops=[Verify; Check; Validate; Run; Display; Emit] |
  emit=CE1c{α=0.000, energy=100.0, matches=6}
}"""


def verify_self():
    """Verify main's own integrity using embedded CE1 seed."""
    # Read the current file to find embedded CE1 seed
    with open(__file__, 'r') as f:
        file_content = f.read()

    # Check if file contains a valid CE1 seed
    if validate_ce1_seed(file_content):
        parsed = parse_ce1_seed(file_content)
        fields = extract_ce1_fields(file_content)

        # Verify expected fields are present
        expected_fields = ['lens', 'mode', 'Ξ']
        missing_fields = [
            field for field in expected_fields if field not in fields]

        if missing_fields:
            print(
                f"⚠️  Self-verification failed: Missing fields {missing_fields}")
            return False
        else:
            print(
                f"✅ Self-verification passed: {len(fields)} fields validated")
            return True
    else:
        print("⚠️  Self-verification failed: No CE1 seed found")
        return False

def main():
    # Self-verify before proceeding
    if not verify_self():
        print("Self-verification failed, but continuing...")

    w = shutil.get_terminal_size().columns
    w = max(60, min(100, w))

    # REPL-friendly display (no effect on print; helpful in interactive)
    echo()

    # 1) responsive menu (subtle)
    items = ['Home', 'Products', 'About', 'Contact']
    space = lambda W, k: max(1, (W - k * 10) // max(1, k - 1))
    menu = responsive(items, spacing=space)
    print(menu(w))

    # 2) labeled symmetry mirror
    sym = symmetry(['◆', '◇'])
    print(mirror_of(sym, w, dusk2))

    # 3) Emergent genomes: deterministic fuse from CLI seed
    arg_seed = ' '.join(sys.argv[1:]) if len(sys.argv) > 1 else ''
    if arg_seed:
        child = fuse(arg_seed)
    else:
        # fallback: simple symmetry with palette context
        child = lambda width: sym(width)
    block_obj = child(w) if callable(child) else child
    block_text = render(block_obj, w)
    print(mark(block_text, style='acetyl', pal=solar))

    # 4) Staged dialog (thin → light → bold) based on width
    thin = lambda width: dialog('Signal', ['forms grow with width'])(width)
    bold = lambda width: dialog('Signal', ['forms grow with width'])(width)
    staged = stage((0, thin), (80, bold))
    print(staged(w))

    # 5) Graceful dict table (metadata)
    meta = {
        'seed': arg_seed or '(none)',
        'palette': getattr(solar, 'label', 'solar'),
        'width': w,
    }
    print(dict_ls(meta, w))


if __name__ == "__main__":
    main()


