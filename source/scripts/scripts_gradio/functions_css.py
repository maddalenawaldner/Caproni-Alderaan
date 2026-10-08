# ------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------CSS------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

body, .gradio-container, .main, .wrap {
    background: #f0f2f7 !important;
    font-family: 'Inter', system-ui, sans-serif !important;
}

footer, .footer, div.footer, .built-with,
.svelte-byatnx, .gradio-footer {
    display: none !important;
}

.gradio-container, .gradio-container * {
    font-size: 1.15rem !important;
    font-family: 'Inter', system-ui, sans-serif !important;
}
label, .label-wrap span {
    font-size: 1.15rem !important;
    font-weight: 500 !important;
    color: #374151 !important;
}
textarea, input, .input-wrap input {
    font-size: 1.15rem !important;
}

/* --- Pagina istruzioni: centrata orizzontalmente e verticalmente --- */
#istruzioni_card {
    width: 100% !important;
    max-width: 100% !important;
    min-height: 90vh !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 40px 20px !important;
    margin: 0 !important;
    box-sizing: border-box !important;
}

#istruzioni_card > * {
    width: 100% !important;
    max-width: 720px !important;
}

#btn_vai_login {
    margin-top: 24px !important;
    max-width: 260px !important;
    align-self: center !important;
}

/* --- Login box: centrato orizzontalmente e verticalmente --- */
#login_card {
    background: transparent !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    border: none !important;
    padding: 40px 20px !important;
    width: 100% !important;
    max-width: 100% !important;
    min-height: 90vh !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    margin: 0 !important;
    box-sizing: border-box !important;
}

#login_card > * {
    width: 100% !important;
    max-width: 560px !important;
}

#login_card > .form,
#login_card .group,
#login_card [data-testid="block"] {
    background: #ffffff !important;
    border-radius: 20px !important;
    border: 1px solid #e2e8f0 !important;
    box-shadow: 0 10px 30px -5px rgba(15, 23, 42, 0.08) !important;
    overflow: hidden !important;
    padding: 24px !important;
}

#login_card .prose,
#login_card [data-testid="markdown"] {
    margin-bottom: 24px !important;
}

#login_card button {
    margin-top: 20px !important;
    padding: 12px !important;
}

#login_card .fieldset {
    border: none !important;
}

#album_col {
    background: #1e2d4a !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 20px rgba(15,23,42,0.18) !important;
    border: none !important;
    padding: 16px 12px !important;
    min-width: 220px !important;
    max-width: 320px !important;
    align-self: flex-start !important;
    margin-right: 20px !important;
    position: sticky !important;
    top: 12px !important;
    max-height: calc(100vh - 24px) !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    flex-wrap: nowrap !important;
}

#album_col > * {
    flex-shrink: 0 !important;
}

#album_col::-webkit-scrollbar { width: 8px; }
#album_col::-webkit-scrollbar-thumb { background: #3b4f75; border-radius: 8px; }
#album_col::-webkit-scrollbar-track { background: transparent; }
#album_col { scrollbar-width: thin; scrollbar-color: #3b4f75 transparent; }

#album_col h3, #album_col .prose h3 {
    color: #f1f5f9 !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    margin-bottom: 10px !important;
}

#album_col .accordion {
    background: #253655 !important;
    border: none !important;
    border-radius: 10px !important;
    margin-bottom: 6px !important;
}

#album_col button,
#album_col .accordion button,
#album_col .accordion span {
    font-size: 1.05rem !important;
    padding: 5px 10px !important;
    color: #475569 !important;
    font-weight: 600 !important;
    background: transparent !important;
    border: none !important;
}

#album_col label span,
#album_col .label-wrap span,
#album_col input[type="radio"] + span {
    font-size: 1.05rem !important;
    color: #475569 !important;
    font-weight: 500 !important;
}

#album_col .wrap {
    flex-direction: column !important;
    gap: 4px !important;
}

#app_header {
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 12px;
    margin-bottom: 28px;
}

#btn_istruzioni_app > button {
    background: transparent !important;
    border: 1px solid #cbd5e1 !important;
    color: #64748b !important;
    font-size: 0.85rem !important;
    padding: 4px 14px !important;
    border-radius: 8px !important;
    box-shadow: none !important;
}

#btn_istruzioni_app > button:hover {
    background: #f1f5f9 !important;
    border-color: #94a3b8 !important;
    color: #374151 !important;
}

#nome_file,
#nome_file > *,
#nome_file .block,
#nome_file .form,
#nome_file .wrap,
#nome_file .wrap-inner,
#nome_file .container {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}
#nome_file {
    border: 1px solid #cbd5e1 !important;
    border-radius: 10px !important;
    padding: 2px 12px !important;
}
#nome_file input, #nome_file textarea {
    text-align: center !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    font-size: 1.2rem !important;
    font-weight: 700 !important;
    color: #1e2d4a !important;
    letter-spacing: 0.01em !important;
    padding: 6px 0 !important;
}

#img_orig .block {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

#img_orig img {
    width: 100% !important;
    height: auto !important;
    max-height: 70vh !important;
    display: block !important;
    object-fit: contain !important;
    border-radius: 10px !important;
}
#img_orig .image-container,
#img_orig [data-testid="image"] {
    height: auto !important;
    max-height: none !important;
}

#mask_viewer {
    padding: 0 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

#ocr_block, #tags_block,
#ocr_block > .block, #tags_block > .block,
#ocr_block.block, #tags_block.block,
#ocr_block > div, #tags_block > div {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    padding: 0 !important;
}
#ocr_block, #tags_block {
    border-top: 1px solid #e2e8f0 !important;
    margin-top: 12px !important;
    padding-top: 12px !important;
}

.tabs > .tab-nav {
    border-bottom: 2px solid #e2e8f0 !important;
    margin-bottom: 16px !important;
}
.tabs > .tab-nav > button {
    font-size: 1.1rem !important;
    font-weight: 500 !important;
    color: #64748b !important;
    border-radius: 0 !important;
    padding: 10px 22px !important;
    transition: color 0.15s ease !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    margin-bottom: -2px !important;
}
.tabs > .tab-nav > button.selected {
    color: #4361ee !important;
    font-weight: 700 !important;
    border-bottom: 2px solid #4361ee !important;
    background: transparent !important;
}

button.primary,
button[data-testid="primary"],
.gradio-button.primary {
    background: #4361ee !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 9px !important;
    border: none !important;
    box-shadow: 0 2px 10px rgba(67,97,238,0.28) !important;
    transition: opacity 0.15s ease, box-shadow 0.15s ease !important;
    font-size: 1.1rem !important;
}
button.primary:hover {
    opacity: 0.88 !important;
    box-shadow: 0 4px 18px rgba(67,97,238,0.38) !important;
}

button.secondary,
button[data-testid="secondary"],
.gradio-button.secondary {
    background: #f1f5f9 !important;
    color: #475569 !important;
    font-weight: 500 !important;
    border-radius: 9px !important;
    border: 1px solid #cbd5e1 !important;
    box-shadow: none !important;
    transition: background 0.15s ease !important;
    font-size: 1.1rem !important;
}
button.secondary:hover {
    background: #e2e8f0 !important;
}

#btn_procedi > button {
    background: #1e2d4a !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    border-radius: 12px !important;
    border: none !important;
    padding: 14px 32px !important;
    box-shadow: 0 4px 20px rgba(15,23,42,0.22) !important;
    transition: opacity 0.15s ease, box-shadow 0.15s ease !important;
    font-size: 1.05rem !important;
}
#btn_procedi > button:hover {
    opacity: 0.85 !important;
    box-shadow: 0 6px 26px rgba(15,23,42,0.30) !important;
}

.multiselect span.token {
    background: #eff6ff !important;
    color: #3b5bdb !important;
    border-radius: 6px !important;
    border: 1px solid #bfdbfe !important;
    font-size: 0.95rem !important;
}

#dd_tags button { display: none !important; }
#dd_tags .token button { display: inline-flex !important; }

#ocr_block .prose p,
#tags_block .prose p {
    font-size: 1.05rem !important;
    margin: 8px 0 0 !important;
    color: #475569 !important;
}

#dialog_unsaved {
    position: fixed !important;
    top: 50% !important;
    left: 50% !important;
    transform: translate(-50%, -50%) !important;
    z-index: 1000 !important;
    width: 90% !important;
    max-width: 360px !important;
    margin: 0 !important;
    padding: 0 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    overflow: visible !important;
}
#dialog_unsaved #dialog_unsaved {
    padding: 22px 24px !important;
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 16px !important;
    box-shadow: 0 20px 50px -10px rgba(15,23,42,0.35), 0 0 0 9999px rgba(15,23,42,0.45) !important;
}
#dialog_unsaved > .block,
#dialog_unsaved .block {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}
#dialog_msg .prose p, #dialog_msg p {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    color: #1e2d4a !important;
    margin: 0 0 18px !important;
    line-height: 1.5 !important;
    text-align: center !important;
}
#dialog_unsaved .row {
    justify-content: center !important;
}
"""