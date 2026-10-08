import base64
import io
import json
import random
from pathlib import Path
from datetime import datetime
import gradio as gr
from PIL import Image
import numpy as np
from config import *
from functions_colonna_sx import *
from functions_mask_tags import *
from functions_jsonl_file import *
from functions_carica_immagini import *
from functions_css import *
from functions_istruzioni_html import *





# ------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------INTERFACCIA------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------

try:
    _theme = gr.themes.Default(primary_hue="blue", neutral_hue="slate")
except Exception:
    _theme = None

_theme_kwarg = {"theme": _theme} if _theme else {}

_highlight_js = """
() => {
    function hl() {
        document.querySelectorAll('#album_col label').forEach(function(label) {
            if (label.textContent.trimStart().startsWith('✓')) {
                label.style.setProperty('background', '#bbf7d0', 'important');
                label.style.setProperty('border-radius', '6px', 'important');
            } else {
                label.style.removeProperty('background');
                label.style.removeProperty('border-radius');
            }
        });
    }
    setInterval(hl, 400);
    return [];
}
"""

with gr.Blocks(title="Revisione immagini", css=custom_css, js=_highlight_js, **_theme_kwarg) as demo:
    gr.HTML('<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">')

    initial_reviewed = get_reviewed_images()

    state_user         = gr.State("")
    state_img_path     = gr.State("")
    state_orig_txt     = gr.State("")
    state_orig_tags    = gr.State([])
    state_reviewed     = gr.State(initial_reviewed)
    state_loaded_txt   = gr.State("")
    state_loaded_tags  = gr.State([])
    state_pending_path = gr.State("")

    # Pagina istruzioni (mostrata per prima)
    with gr.Group(elem_id="istruzioni_card") as grp_istruzioni:
        gr.HTML(ISTRUZIONI_HTML)
        btn_vai_login = gr.Button("Ho capito, inizia →", variant="primary", elem_id="btn_vai_login")

    with gr.Group(elem_id="login_card", visible=False) as grp_login:
        gr.HTML('<h1 style="font-family:\'Inter\',sans-serif;color:#1e2d4a;font-size:1.9rem;font-weight:700;margin:0 0 6px;letter-spacing:-0.01em;">Progetto Caproni</h1><p style="font-family:\'Inter\',sans-serif;color:#64748b;margin:0 0 28px;font-size:1.05rem;">Inserisci i tuoi dati per accedere alla revisione.</p>')
        with gr.Row():
            inp_nome  = gr.Textbox(label="Nome e Cognome *", placeholder="Es. Mario Rossi")
            inp_ruolo = gr.Textbox(label="Ruolo (opzionale)", placeholder="Es. Ricercatore")
        btn_login     = gr.Button("Accedi →", variant="primary")
        msg_login_err = gr.Markdown(visible=False)

    with gr.Group(visible=False) as grp_app:
        with gr.Row(elem_id="app_header"):
            lbl_utente = gr.Markdown("", scale=10)
            btn_istruzioni_app = gr.Button("Istruzioni", elem_id="btn_istruzioni_app", scale=0, size="sm")

        # Modale istruzioni: parte vuota, viene popolata al click del pulsante
        modal_html = gr.HTML("")

        with gr.Row():

            with gr.Column(scale=3, min_width=220, elem_id="album_col"):
                gr.Markdown("### Album")
                radios = []
                accordions = []
                for aname in album_names_sorted:
                    with gr.Accordion(aname, open=False) as acc:
                        init_choices = build_radio_choices(initial_reviewed, aname)
                        r = gr.Radio(choices=init_choices, label="Seleziona un'immagine", interactive=True)
                        radios.append(r)
                    accordions.append(acc)

            with gr.Column(scale=8):

                with gr.Group(visible=False, elem_id="dialog_unsaved") as dialog_unsaved:
                    lbl_dialog_msg = gr.Markdown("", elem_id="dialog_msg")
                    with gr.Row():
                        btn_dialog_salva   = gr.Button("💾 Salva e procedi", variant="primary")
                        btn_dialog_annulla = gr.Button("✏️ Continua a modificare")

                txt_nome = gr.Textbox(value="Seleziona un'immagine dall'elenco di sinistra", label="", show_label=False, interactive=False, elem_id="nome_file", container=False)

                gr.HTML('<div style="height: 12px;"></div>')

                with gr.Tabs(selected=0) as tabs_main:

                    with gr.TabItem("📄  Testo OCR", id=0):
                        img_orig = gr.Image(
                            value=None,
                            type="filepath",
                            label="Immagine originale",
                            interactive=False,
                            elem_id="img_orig",
                            container=False,
                        )
                        with gr.Group(elem_id="ocr_block"):
                            gr.HTML('<p style="font-family:\'Inter\',sans-serif;color:#64748b;font-size:0.82rem;font-weight:700;margin:0 0 12px;letter-spacing:0.10em;text-transform:uppercase;">Testo OCR <span style="font-size:0.78rem;font-weight:400;letter-spacing:0;text-transform:none;color:#94a3b8;">— Aggiungi e/o modifica il testo</span></p>')
                            txt_ocr = gr.Textbox(
                                label="", show_label=False, lines=4, interactive=True,
                                placeholder="",
                            )
                            with gr.Row():
                                btn_salva_ocr   = gr.Button("💾 Salva testo", variant="primary", scale=1)
                                btn_annulla_ocr = gr.Button("↩ Annulla", scale=1)
                            lbl_stato_ocr = gr.Markdown("")

                    with gr.TabItem("🏷️  Tag e maschere", id=1):
                        mask_viewer = gr.HTML("", elem_id="mask_viewer")
                        with gr.Group(elem_id="tags_block"):
                            gr.HTML('<p style="font-family:\'Inter\',sans-serif;color:#64748b;font-size:0.82rem;font-weight:700;margin:0 0 12px;letter-spacing:0.10em;text-transform:uppercase;">Tag <span style="font-size:0.78rem;font-weight:400;letter-spacing:0;text-transform:none;color:#94a3b8;">— Digita un tag e premi Invio per aggiungerlo; clicca × per rimuoverlo.</span></p>')
                            dd_tags = gr.Dropdown(
                                choices=[], value=[],
                                multiselect=True,
                                allow_custom_value=True,
                                label="", show_label=False,
                                interactive=True,
                                elem_id="dd_tags",
                            )
                            with gr.Row():
                                btn_salva_tags   = gr.Button("💾 Salva tag e procedi →", variant="primary", scale=1)
                                btn_annulla_tags = gr.Button("↩ Annulla", scale=1)
                            lbl_stato_tags = gr.Markdown("")


# ------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------CALLBACKS--------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------


    btn_vai_login.click(
        fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
        inputs=[],
        outputs=[grp_istruzioni, grp_login],
    )

    btn_istruzioni_app.click(
        fn=lambda: _MODAL_HTML,
        inputs=[],
        outputs=[modal_html],
    )

    def do_login(nome, ruolo):
        if not nome.strip():
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                "",
                gr.update(visible=True, value="⚠️ Il campo *Nome e Cognome* è obbligatorio."),
            )
        user = nome.strip() + (f" ({ruolo.strip()})" if ruolo.strip() else "")
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            user,
            gr.update(visible=False, value=""),
        )

    _login_kwargs = dict(
        fn=do_login,
        inputs=[inp_nome, inp_ruolo],
        outputs=[grp_login, grp_app, state_user, msg_login_err],
    )
    _then_kwargs = dict(
        fn=lambda u: f"👤 **{u}**",
        inputs=[state_user],
        outputs=[lbl_utente],
    )

    btn_login.click(**_login_kwargs).then(**_then_kwargs)
    inp_nome.submit(**_login_kwargs).then(**_then_kwargs)
    inp_ruolo.submit(**_login_kwargs).then(**_then_kwargs)

    _RADIO_OUTPUTS = (
        [img_orig, mask_viewer, txt_nome, dd_tags, txt_ocr,
         state_img_path, state_orig_txt, state_orig_tags,
         lbl_stato_ocr, lbl_stato_tags, tabs_main]
        + radios + accordions
        + [dialog_unsaved, lbl_dialog_msg, state_pending_path,
           state_loaded_txt, state_loaded_tags]
    )

    for i, r in enumerate(radios):
        def make_fn(radio_idx):
            def fn(path, curr_txt, curr_tags, loaded_txt, loaded_tags, curr_img_path):
                n_acc = len(accordions)
                n_rad = len(radios)
                no_op = [gr.update()] * len(_RADIO_OUTPUTS)

                if path is None or path == curr_img_path:
                    return no_op

                txt_mod  = (curr_txt  or "").strip() != (loaded_txt  or "").strip()
                tags_mod = sorted(curr_tags or []) != sorted(loaded_tags or [])
                if curr_img_path and (txt_mod or tags_mod):
                    msg = "Salvare le modifiche effettuate?"
                    curr_album = find_album_for_image(curr_img_path)
                    curr_album_idx = album_names_sorted.index(curr_album) if curr_album in album_names_sorted else None
                    radio_upds = []
                    for j in range(n_rad):
                        if j == curr_album_idx:
                            radio_upds.append(gr.update(value=curr_img_path))
                        elif j == radio_idx:
                            radio_upds.append(gr.update(value=None))
                        else:
                            radio_upds.append(gr.update())
                    return (
                        [gr.update()] * 11
                        + radio_upds
                        + [gr.update()] * n_acc
                        + [gr.update(visible=True), msg, path, gr.update(), gr.update()]
                    )

                _mask_orig, nome, txt, tags, has_masks, has_text = load_image_data(path)
                annulla_txt, annulla_tags = load_annulla_target(Path(path).stem)
                overlays_b64 = get_overlay_b64(Path(path).stem, path)
                html = make_mask_viewer_html(path, overlays_b64)
                album = find_album_for_image(path)
                ocr_msg  = "" if has_text  else "⚠️ Non è stato identificato alcun testo"
                tags_msg = "" if has_masks else "⚠️ Non sono stati identificati alcun tag o maschere"
                base = [
                    path, html, nome,
                    gr.update(choices=tags, value=tags),
                    txt, path, annulla_txt, annulla_tags, ocr_msg, tags_msg,
                    gr.update(selected=0),
                ]
                radio_upds = [
                    gr.update() if j == radio_idx else None
                    for j in range(n_rad)
                ]
                acc_upds = [gr.update(open=(aname == album)) for aname in album_names_sorted]
                return (
                    base + radio_upds + acc_upds
                    + [gr.update(visible=False), "", "", txt, tags]
                )
            return fn

        r.change(
            fn=make_fn(i),
            inputs=[r, txt_ocr, dd_tags, state_loaded_txt, state_loaded_tags, state_img_path],
            outputs=_RADIO_OUTPUTS,
        )

    def do_salva_ocr(img_path):
        if not img_path:
            return "⚠️ Analizza prima un'immagine.", gr.update()
        return "", gr.update(selected=1)

    btn_salva_ocr.click(
        fn=do_salva_ocr,
        inputs=[state_img_path],
        outputs=[lbl_stato_ocr, tabs_main],
    )

    btn_annulla_ocr.click(fn=lambda o: o, inputs=[state_orig_txt], outputs=[txt_ocr])

    def _load_and_navigate(next_path, new_rev):
        """Carica next_path e costruisce il blocco di aggiornamenti comuni."""
        _mask_orig, nome, next_txt, next_tags, has_masks, has_text = load_image_data(next_path)
        annulla_txt, annulla_tags = load_annulla_target(Path(next_path).stem)
        next_name = Path(next_path).stem
        overlays_b64 = get_overlay_b64(next_name, next_path)
        html = make_mask_viewer_html(next_path, overlays_b64)
        next_album = find_album_for_image(next_path)
        ocr_msg  = "" if has_text  else "⚠️ Non è stato identificato alcun testo"
        tags_msg = "" if has_masks else "⚠️ Non è stata identificata nessun tag e maschera"
        radio_upds = []
        for aname in album_names_sorted:
            choices = build_radio_choices(new_rev, aname)
            value   = next_path if aname == next_album else None
            radio_upds.append(gr.update(choices=choices, value=value))
        acc_upds = [gr.update(open=(aname == next_album)) for aname in album_names_sorted]
        return (
            next_path, html, nome, next_txt,
            gr.update(choices=next_tags, value=next_tags),
            next_path, annulla_txt, annulla_tags,
            ocr_msg, tags_msg, new_rev, gr.update(selected=0),
            next_txt, next_tags,
        ), tuple(radio_upds), tuple(acc_upds)

    def do_salva_tags(tags, img_path, txt, user, reviewed):
        if not img_path:
            return (
                (gr.update(),) * 9
                + ("⚠️ Analizza prima un'immagine.",)
                + (reviewed, gr.update(), gr.update(), gr.update())
                + (gr.update(),) * len(radios)
                + (gr.update(),) * len(accordions)
            )
        name = Path(img_path).stem
        save_revision(name, img_path, find_mask_path(name), txt, tags or [], user)
        new_rev = set(reviewed) | {name}
        all_paths = [str(p) for p in lista_path_img]
        try:
            idx = all_paths.index(img_path)
        except ValueError:
            idx = -1
        next_path = all_paths[(idx + 1) % len(all_paths)]
        base, radio_upds, acc_upds = _load_and_navigate(next_path, new_rev)
        return base + radio_upds + acc_upds

    btn_salva_tags.click(
        fn=do_salva_tags,
        inputs=[dd_tags, state_img_path, txt_ocr, state_user, state_reviewed],
        outputs=[
            img_orig, mask_viewer, txt_nome, txt_ocr, dd_tags,
            state_img_path, state_orig_txt, state_orig_tags,
            lbl_stato_ocr, lbl_stato_tags,
            state_reviewed, tabs_main,
            state_loaded_txt, state_loaded_tags,
        ] + radios + accordions,
    )

    btn_dialog_annulla.click(
        fn=lambda: (gr.update(visible=False), ""),
        inputs=[],
        outputs=[dialog_unsaved, state_pending_path],
    )

    def do_dialog_salva(tags, img_path, txt, user, reviewed, pending_path):
        if not img_path or not pending_path:
            return (gr.update(visible=False), "") + (gr.update(),) * (
                14 + len(radios) + len(accordions)
            )
        name = Path(img_path).stem
        save_revision(name, img_path, find_mask_path(name), txt, tags or [], user)
        new_rev = set(reviewed) | {name}
        base, radio_upds, acc_upds = _load_and_navigate(pending_path, new_rev)
        return (gr.update(visible=False), "") + base + radio_upds + acc_upds

    btn_dialog_salva.click(
        fn=do_dialog_salva,
        inputs=[dd_tags, state_img_path, txt_ocr, state_user, state_reviewed, state_pending_path],
        outputs=[
            dialog_unsaved, state_pending_path,
            img_orig, mask_viewer, txt_nome, txt_ocr, dd_tags,
            state_img_path, state_orig_txt, state_orig_tags,
            lbl_stato_ocr, lbl_stato_tags,
            state_reviewed, tabs_main,
            state_loaded_txt, state_loaded_tags,
        ] + radios + accordions,
    )

    btn_annulla_tags.click(
        fn=lambda orig: gr.update(choices=orig, value=orig),
        inputs=[state_orig_tags],
        outputs=[dd_tags],
    )



demo.launch(inbrowser=True, allowed_paths=[str(path_album), str(path_maschere_singole)])
