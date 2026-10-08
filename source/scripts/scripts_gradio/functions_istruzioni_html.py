
# ------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------ISTRUZIONI-------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------

# Card condivisa (senza titolo né sottotitolo)
CARD_HTML = """
<div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:20px;
            box-shadow:0 10px 30px -5px rgba(15,23,42,0.08);padding:28px 32px;
            font-family:'Inter',system-ui,sans-serif;">
  <h2 style="color:#1e2d4a;font-size:1.1rem;font-weight:700;margin:0 0 16px;
             letter-spacing:0.06em;text-transform:uppercase;">Come funziona</h2>
  <ol style="color:#374151;font-size:1.05rem;line-height:1.8;margin:0 0 24px;padding-left:20px;">
    <li>Esegui il login inserendo nome e ruolo.</li>
    <li>Seleziona un album dalla colonna di sinistra e scegli un'immagine.</li>
    <li>Nella scheda <strong>Testo OCR</strong>: controlla e correggi il testo riconosciuto automaticamente. Clicca <em>Salva testo</em> per passare alla scheda successiva.</li>
    <li>Nella scheda <strong>Tag e maschere</strong>: verifica i tag e le maschere di segmentazione. Aggiungi o rimuovi tag se necessario. Rimuovendo un tag verrà rimossa la corrispondente maschera. Aggiungendo un tag <em style="color:#94a3b8;">[da completare]</em>.</li>
    <li>Clicca <em>Salva tag e procedi</em> per salvare la revisione e passare all'immagine successiva.</li>
  </ol>
  <h2 style="color:#1e2d4a;font-size:1.1rem;font-weight:700;margin:0 0 16px;
             letter-spacing:0.06em;text-transform:uppercase;">Note</h2>
  <ul style="color:#374151;font-size:1.05rem;line-height:1.8;margin:0;padding-left:20px;">
    <li>I file già revisionati appaiono con ✓ e sfondo verde nella colonna di sinistra.</li>
    <li>Se modifichi testo o tag e selezioni un'altra immagine prima di salvare, comparirà un avviso.</li>
    <li>Puoi ingrandire le immagini con il pulsante di schermo intero.</li>
  </ul>
</div>
"""

# Pagina intro: titolo + sottotitolo + card
ISTRUZIONI_HTML = (
    '<div style="font-family:\'Inter\',system-ui,sans-serif;width:100%;">'
    '<h1 style="color:#1e2d4a;font-size:3.2rem;font-weight:700;margin:0 0 6px;'
    'letter-spacing:-0.02em;text-align:center;">Progetto Caproni</h1>'
    '<p style="color:#64748b;font-size:1.05rem;margin:0 0 28px;text-align:center;">'
    'Istruzioni per la revisione del materiale fotografico.</p>'
    + CARD_HTML +
    '</div>'
)

# Modale: solo la card, senza titolo né sottotitolo
MODAL_HTML = (
    '<div id="istruzioni_overlay"'
    ' onclick="if(event.target===this)this.style.display=\'none\'"'
    ' style="position:fixed;top:0;left:0;width:100vw;height:100vh;'
    'background:rgba(15,23,42,0.92);z-index:9999;display:flex;'
    'align-items:center;justify-content:center;padding:40px 20px;overflow-y:auto;">'
    '<div style="max-width:680px;width:100%;margin:0 auto;">'
    + CARD_HTML +
    '<div style="display:flex;justify-content:center;margin-top:20px;">'
    '<button'
    ' onclick="document.getElementById(\'istruzioni_overlay\').style.display=\'none\'"'
    ' style="background:#1e2d4a;color:#ffffff;border:none;padding:10px 28px;'
    'border-radius:8px;font-size:0.95rem;cursor:pointer;font-family:Inter,sans-serif;">'
    '✕ Chiudi</button></div>'
    '</div>'
    '</div>'
)
