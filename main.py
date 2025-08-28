def main():
    from sprixel import responsive, reflect_wave, dialog, sea, dusk, solar, squeeze, fuse
    from ops import symmetry, stage, mark, promote
    from sprixel2 import echo, mirror_of, dusk as dusk2, dict_ls, render
    import sys
    import shutil

    try:
        w = shutil.get_terminal_size().columns
    except Exception:
        w = 80
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

    # compact palette bars removed for subtlety


if __name__ == "__main__":
    main()


