#!/usr/bin/env python3
"""
audiobook_translate_chunked.py

Objetivo:
- Extrai texto de .txt/.pdf/.epub ou aceita texto manual.
- Detecta idioma.
- Pergunta se quer traduzir; se sim, pergunta idioma alvo.
- Tradução feita em blocos/chunks (para evitar limite ~5000 chars do GoogleTranslator).
  - Normaliza alguns caracteres 'mojibake' comuns antes de traduzir.
  - Retry por bloco com backoff.
- Depois da tradução (ou sem tradução), pergunta qual VOZ usar (sugestão por idioma final).
- Gera partes com edge-tts e junta em book_final.mp3.
- Opcional: gera MP4 com MoviePy.
"""
import asyncio
import os
import re
import random
import shutil
import subprocess
import sys
import zipfile
import time
from pathlib import Path
from typing import Optional
from contextlib import redirect_stdout, redirect_stderr, contextmanager

# ----------------- UTIL -----------------
@contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield

def yesno(prompt: str, default: bool = False) -> bool:
    yes = {'y', 'yes', 's', 'sim'}
    no = {'n', 'no', 'nao', 'não'}
    if default:
        prompt_full = f"{prompt} [Y/n]: "
    else:
        prompt_full = f"{prompt} [y/N]: "
    while True:
        ans = input(prompt_full).strip().lower()
        if ans == '' and default is not None:
            return default
        if ans in yes:
            return True
        if ans in no:
            return False
        print("❌ Responda sim/nao (s/n) ou y/n.")

def clear_console():
    try:
        os.system('cls' if os.name == 'nt' else 'clear')
    except Exception:
        pass

def _progress_bar_str(curr: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return ""
    frac = curr / total
    filled = int(round(frac * width))
    bar = "█" * filled + "-" * (width - filled)
    pct = frac * 100
    return f"[{bar}] {curr}/{total} ({pct:5.1f}%)"

def update_progress_line(curr: int, total: int, status: str = ""):
    bar = _progress_bar_str(curr, total)
    status_part = f" {status}" if status else ""
    line = f"\r  {bar}  [{curr}/{total}]{status_part}".ljust(120)
    sys.stdout.write(line)
    sys.stdout.flush()

# ----------------- CONFIG -----------------
DEFAULT_VOICE_PT = "pt-BR-FranciscaNeural"
MAX_CHARS = 4200             # para split em partes de TTS
MAX_TRANSLATE_CHARS = 4500   # para enviar ao tradutor (safety margin < 5000)
TRANSLATE_RETRIES = 4
TRANSLATE_BACKOFF = 1.0
DELAY_BETWEEN = (0.6, 1.2)
MAX_RETRIES = 6
INITIAL_BACKOFF = 1.0

SUPPORTED_EXTS = ['.pdf', '.txt', '.md', '.epub']
SUPPORTED_IMG_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}

VOICE_LIST_BY_SHORT = {
    'pt': ['pt-BR-FranciscaNeural', 'pt-BR-HeloisaNeural', 'pt-BR-AntonioNeural'],
    'en': ['en-US-GuyNeural', 'en-US-JennyNeural', 'en-US-AriaNeural'],
    'es': ['es-ES-AlvaroNeural', 'es-ES-ElviraNeural'],
    'fr': ['fr-FR-DeniseNeural', 'fr-FR-HenriNeural'],
    'de': ['de-DE-ConradNeural', 'de-DE-AmalaNeural'],
    'it': ['it-IT-ElsaNeural'],
    'ja': ['ja-JP-NanamiNeural'],
    'ko': ['ko-KR-SunHiNeural'],
    'zh-cn': ['zh-CN-XiaoxiaoNeural'],
}

DETECTED_LANG_TO_VOICE = {
    'pt': 'pt-BR-FranciscaNeural',
    'en': 'en-US-GuyNeural',
    'es': 'es-ES-ElviraNeural',
    'fr': 'fr-FR-DeniseNeural',
    'de': 'de-DE-ConradNeural',
    'it': 'it-IT-ElsaNeural',
    'ja': 'ja-JP-NanamiNeural',
    'ko': 'ko-KR-SunHiNeural',
    'zh-cn': 'zh-CN-XiaoxiaoNeural'
}

# ----------------- DEPENDÊNCIAS -----------------
try:
    import edge_tts
    from aiohttp import client_exceptions
except Exception:
    print("❌ Instale 'edge-tts' primeiro: python -m pip install edge-tts")
    sys.exit(1)

try:
    from moviepy.editor import AudioFileClip, ImageClip, ColorClip
except Exception:
    AudioFileClip = ImageClip = ColorClip = None

# PDF / EPUB libs
_PDF_LIB = None
try:
    from PyPDF2 import PdfReader
    _PDF_LIB = "pypdf2"
except Exception:
    try:
        import pdfplumber
        _PDF_LIB = "pdfplumber"
    except Exception:
        _PDF_LIB = None

_EPUB_LIB = None
_BS4 = False
try:
    import ebooklib
    from ebooklib import epub
    _EPUB_LIB = "ebooklib"
except Exception:
    _EPUB_LIB = None
try:
    from bs4 import BeautifulSoup
    _BS4 = True
except Exception:
    _BS4 = False

# ----------------- I/O e EXTRAÇÃO -----------------
def safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding='utf-8')
    except Exception:
        try:
            return path.read_text(encoding='cp1252')
        except Exception as e:
            raise RuntimeError(f"Could not read file '{path}': {e}")

def extract_text_from_pdf(path: Path) -> str:
    if _PDF_LIB == "pypdf2":
        reader = PdfReader(str(path))
        texts = []
        for page in reader.pages:
            try:
                t = page.extract_text()
            except Exception:
                t = None
            texts.append(t or "")
        return "\n".join(texts)
    elif _PDF_LIB == "pdfplumber":
        import pdfplumber
        texts = []
        with pdfplumber.open(str(path)) as pdf:
            for p in pdf.pages:
                texts.append(p.extract_text() or "")
        return "\n".join(texts)
    else:
        raise RuntimeError("No PDF library found. Install PyPDF2 or pdfplumber.")

def extract_text_from_epub(path: Path) -> str:
    texts = []
    if _EPUB_LIB == "ebooklib":
        book = epub.read_epub(str(path))
        items = []
        for item in book.get_items():
            name = getattr(item, "get_name", lambda: "")()
            if isinstance(name, str) and name.lower().endswith(('.html', '.xhtml', '.htm', '.xht')):
                items.append(item)
            else:
                try:
                    mt = getattr(item, "media_type", "")
                    if mt and ("html" in mt or "xhtml" in mt):
                        items.append(item)
                except Exception:
                    pass
        if not items:
            items = list(book.get_items())
        for item in items:
            try:
                content = item.get_content()
            except Exception:
                try:
                    content = item.get_body_content()
                except Exception:
                    content = None
            if not content:
                continue
            if isinstance(content, bytes):
                content = content.decode('utf-8', errors='ignore')
            if _BS4:
                soup = BeautifulSoup(content, 'html.parser')
                texts.append(soup.get_text(separator='\n'))
            else:
                t = re.sub(r'<[^>]+>', '', content)
                texts.append(t)
        return '\n'.join(t for t in texts if t)
    # fallback zip
    with zipfile.ZipFile(str(path), 'r') as z:
        html_names = [n for n in z.namelist() if n.lower().endswith(('.html', '.xhtml', '.htm'))]
        for name in html_names:
            raw = z.read(name).decode('utf-8', errors='ignore')
            if _BS4:
                soup = BeautifulSoup(raw, 'html.parser')
                texts.append(soup.get_text(separator='\n'))
            else:
                t = re.sub(r'<[^>]+>', '', raw)
                texts.append(t)
    return '\n'.join(t for t in texts if t)

def list_supported_files_in_cwd(cwd: Path):
    files = [p for p in cwd.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    return sorted(files, key=lambda p: p.name.lower())

# ----------------- SPLIT / NORMALIZAÇÃO -----------------
def split_text_preserve_sentences(text: str, limit: int) -> list:
    """Split preserving sentences, fallback to hard cut if necessary."""
    parts = []
    text = text.strip()
    while text:
        if len(text) <= limit:
            parts.append(text.strip())
            break
        # try cut at punctuation
        candidates = [text.rfind(p, 0, limit) for p in ('.', '!', '?', '\n', ';', ',')]
        cut = max(candidates)
        if cut <= 0:
            cut = limit
        # prevent infinite loop
        part = text[:cut].strip()
        if not part:
            part = text[:limit].strip()
            text = text[limit:].lstrip()
        else:
            text = text[cut:].lstrip()
        parts.append(part)
    return parts

def normalize_mojibake(s: str) -> str:
    """Fix common mojibake sequences (e.g. â€™) and trim excessive whitespace."""
    if not s:
        return s
    # common replacements
    repl = {
        "â€™": "’",
        "â€'": "'",
        "â€œ": "“",
        "â€�": "”",
        "â€“": "–",
        "â€”": "—",
        "â€¦": "…",
        "\u2018": "'",
        "\u2019": "’",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    # try re-encoding heuristic if those sequences are frequent
    if "Ã" in s or "â" in s:
        try:
            s2 = s.encode('latin1').decode('utf-8')
            # if result looks better (fewer weird sequences), keep it
            if s2.count('â') < s.count('â'):
                s = s2
        except Exception:
            pass
    # normalize whitespace
    s = re.sub(r'[ \t]+', ' ', s)
    s = re.sub(r'\n{3,}', '\n\n', s)
    return s.strip()

# ----------------- TRANSLATION EM BLOCOS -----------------
def safe_translate_text_chunks(text: str, target: str, source: Optional[str] = None) -> Optional[str]:
    """
    Translate `text` to `target` in chunks <= MAX_TRANSLATE_CHARS.
    Returns translated text or None if failed and user chose to abort.
    """
    try:
        from deep_translator import GoogleTranslator as GT
    except Exception:
        print("⚠️ deep-translator não instalado. Instale com: python -m pip install deep-translator")
        return None

    # normalize whole text first
    text = normalize_mojibake(text)

    # split into paragraphs to keep structure and avoid breaking across logical paragraphs
    paragraphs = text.split('\n\n')
    chunks = []
    for p in paragraphs:
        p = p.strip()
        if not p:
            chunks.append("")  # preserve blank paragraph
            continue
        if len(p) <= MAX_TRANSLATE_CHARS:
            chunks.append(p)
        else:
            # split long paragraph into sub-chunks
            subparts = split_text_preserve_sentences(p, MAX_TRANSLATE_CHARS)
            chunks.extend(subparts)

    translated_parts = []
    total = len(chunks)
    print(f"🔁 Traduzindo em {total} bloco(s)...")
    for idx, chunk in enumerate(chunks, start=1):
        # preserve blank paragraph
        if not chunk:
            translated_parts.append("")
            continue

        ok = False
        attempt = 0
        while not ok and attempt < TRANSLATE_RETRIES:
            attempt += 1
            try:
                with suppress_output():
                    src = source if source and source != 'unknown' else 'auto'
                    translator = GT(source=src, target=target)
                    translated = translator.translate(chunk)
                # small normalization in result
                translated = normalize_mojibake(translated)
                translated_parts.append(translated)
                ok = True
            except Exception as e:
                wait = TRANSLATE_BACKOFF * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                print(f"\n  ⚠️ Erro ao traduzir bloco {idx}/{total} (tentativa {attempt}/{TRANSLATE_RETRIES}): {e}")
                if attempt < TRANSLATE_RETRIES:
                    print(f"    ...aguardando {wait:.1f}s e tentando novamente...")
                    time.sleep(wait)
                else:
                    # ultima tentativa falhou
                    print(f"    ❌ Bloco {idx} falhou depois de {TRANSLATE_RETRIES} tentativas.")
                    # tentar dividir este bloco em pedaços menores (fallback)
                    if len(chunk) > 1000:
                        print("    → Tentando dividir bloco em pedaços menores e traduzir cada um...")
                        smaller = split_text_preserve_sentences(chunk, max(1000, MAX_TRANSLATE_CHARS // 3))
                        inner_ok = True
                        inner_translated = []
                        for j, sub in enumerate(smaller, start=1):
                            sub_ok = False
                            sub_attempt = 0
                            while not sub_ok and sub_attempt < 3:
                                sub_attempt += 1
                                try:
                                    with suppress_output():
                                        translator = GT(source=src, target=target)
                                        tsub = translator.translate(sub)
                                    inner_translated.append(normalize_mojibake(tsub))
                                    sub_ok = True
                                except Exception:
                                    time.sleep(0.5 * sub_attempt)
                                    continue
                            if not sub_ok:
                                inner_ok = False
                                break
                        if inner_ok:
                            translated_parts.append("\n".join(inner_translated))
                            ok = True
                            break
                    # se não conseguiu, pergunta ao usuário se quer continuar sem traduzir este bloco
                    if yesno("Deseja continuar SEM traduzir este bloco e seguir? (s/N):", default=False):
                        translated_parts.append(chunk)  # keep original
                        ok = True
                        break
                    else:
                        print("Operação cancelada pelo usuário durante tradução.")
                        return None
        # show progress
        update_progress_line(idx, total, "traduzido" if ok else "falha")
    print()  # newline after progress
    # re-join preserving double newlines between paragraphs
    # Some translated_parts items may contain internal newlines; we recombine by paragraphs
    result = "\n\n".join(translated_parts)
    return result

# ----------------- TTS with retries -----------------
async def save_part_with_retries(text_part: str, filename: str, voice: str) -> bool:
    backoff = INITIAL_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            communicator = edge_tts.Communicate(text_part, voice=voice)
            with suppress_output():
                await communicator.save(filename)
            return True
        except client_exceptions.WSServerHandshakeError as e:
            status = getattr(e, 'status', None)
            print(f"\n  → ⚠️ WSS Error (status={status}) attempt {attempt}/{MAX_RETRIES} for {filename}: {e}")
        except Exception as e:
            print(f"\n  → ❌ Error (attempt {attempt}/{MAX_RETRIES}) for {filename}: {e}")
        if attempt < MAX_RETRIES:
            wait = backoff + random.uniform(0, 0.5)
            print(f"  ...waiting {wait:.1f}s before retrying...")
            await asyncio.sleep(wait)
            backoff *= 2
    print(f"\n  ❌ Failed after {MAX_RETRIES} attempts: {filename}")
    return False

# ----------------- MAIN FLOW -----------------
async def generate_and_join():
    clear_console()
    print(" 🔊  Audiobook: traduzir em blocos (chunked) e gerar TTS  🔊 ".center(80, "="))
    cwd = Path(".").resolve()
    print(f"📁 Pasta atual: {cwd}\n")

    # prompt para arquivos / texto
    files = list_supported_files_in_cwd(cwd)
    print('\n🔎 Supported file types: ' + ', '.join(SUPPORTED_EXTS))
    if files:
        print('\n📖 Arquivos encontrados:')
        for i, p in enumerate(files, start=1):
            print(f"  {i}) {p.name}")
    else:
        print('\nNenhum arquivo suportado encontrado no diretório atual.')

    print("\nEscolha arquivo(s) por número (ex: 1 ou 1,3) ou 'm' para colar texto, 'p' para caminho, 'a' para todos, 'q' para sair.")
    while True:
        choice = input("➡️ Seleção: ").strip().lower()
        if not choice:
            continue
        if choice in ('q', 'quit', 'exit'):
            print("Saindo...")
            return
        if choice == 'm':
            print("Cole o texto. Digite EOF em uma linha para finalizar.")
            lines = []
            while True:
                try:
                    line = input()
                except EOFError:
                    break
                if line.strip().upper() == 'EOF':
                    break
                lines.append(line)
            text = "\n".join(lines).strip()
            base_name = 'manual_input'
            break
        if choice == 'p':
            path_input = input("Caminho do arquivo: ").strip()
            p = Path(path_input)
            if not p.exists() or not p.is_file():
                print("Arquivo não encontrado. Tente novamente.")
                continue
            if p.suffix.lower() not in SUPPORTED_EXTS:
                print("Extensão não suportada.")
                continue
            if p.suffix.lower() == '.pdf':
                text = extract_text_from_pdf(p)
            elif p.suffix.lower() == '.epub':
                text = extract_text_from_epub(p)
            else:
                text = safe_read_text(p)
            base_name = p.stem
            break
        if choice == 'a':
            # combine all files
            combined = []
            for p in files:
                if p.suffix.lower() == '.pdf':
                    combined.append(extract_text_from_pdf(p))
                elif p.suffix.lower() == '.epub':
                    combined.append(extract_text_from_epub(p))
                else:
                    combined.append(safe_read_text(p))
            text = "\n\n".join(combined)
            base_name = "combined"
            break
        # else numbers
        nums = [s.strip() for s in choice.split(',') if s.strip()]
        if not nums:
            print("Seleção inválida.")
            continue
        ok = True
        selected = []
        for n in nums:
            if not n.isdigit():
                ok = False
                break
            idx = int(n)
            if idx < 1 or idx > len(files):
                ok = False
                break
            selected.append(files[idx - 1])
        if not ok:
            print("Seleção inválida.")
            continue
        # read selected files
        combined = []
        for p in selected:
            if p.suffix.lower() == '.pdf':
                combined.append(extract_text_from_pdf(p))
            elif p.suffix.lower() == '.epub':
                combined.append(extract_text_from_epub(p))
            else:
                combined.append(safe_read_text(p))
        text = "\n\n".join(combined)
        base_name = selected[0].stem if len(selected) == 1 else "combined"
        break

    if not text or not text.strip():
        print("Texto vazio. Saindo.")
        return

    # detect language
    print("\n🌐 Detectando idioma...")
    detected_lang = None
    detector_used = None
    confidence = None
    sample = text[:20000] if len(text) > 20000 else text
    try:
        with suppress_output():
            import langid
            code, score = langid.classify(sample)
        detected_lang = code
        confidence = float(score)
        detector_used = 'langid'
    except Exception:
        try:
            from langdetect import detect_langs
            with suppress_output():
                probs = detect_langs(sample)
            if probs:
                best = probs[0]
                detected_lang = best.lang
                try:
                    confidence = float(best.prob)
                except Exception:
                    confidence = None
                detector_used = 'langdetect'
        except Exception:
            # heuristic
            s = sample.lower()
            stopwords = {
                'pt': [' que ', ' de ', ' e ', ' o '],
                'en': [' the ', ' and ', ' to ', ' of '],
                'es': [' que ', ' de ', ' y '],
            }
            scores = {k: sum(s.count(w) for w in words) for k, words in stopwords.items()}
            best_code, best_count = max(scores.items(), key=lambda x: x[1])
            detected_lang = best_code if best_count > 0 else 'unknown'
            detector_used = 'heuristic'
    if detected_lang and detected_lang.startswith('zh'):
        detected_lang = 'zh-cn'
    print(f"🔎 Detectado: {detected_lang} (método: {detector_used})" + (f" — confiança: {confidence:.2f}" if confidence else ""))

    # perguntar se quer traduzir
    do_translate = yesno("\n🌐 Deseja traduzir o texto para outro idioma antes de gerar o áudio?", default=False)
    target_lang = None
    if do_translate:
        print("\nEscolha o idioma alvo:")
        options = {
            "1": "pt",
            "2": "en",
            "3": "es",
            "4": "fr",
            "5": "de",
            "6": "it",
            "7": "ja",
            "8": "ko",
            "9": "zh-cn"
        }
        print(" 1) Português (pt)\n 2) Inglês (en)\n 3) Espanhol (es)\n 4) Francês (fr)\n 5) Alemão (de)\n 6) Italiano (it)\n 7) Japonês (ja)\n 8) Coreano (ko)\n 9) Chinês (zh-cn)")
        sel = input("Número: ").strip()
        target_lang = options.get(sel)
        if not target_lang:
            print("Opção inválida. Pulando tradução.")
            do_translate = False

    if do_translate and target_lang:
        translated = safe_translate_text_chunks(text, target_lang, source=detected_lang)
        if translated is None:
            # tradução falhou e usuário cancelou ou deep-translator não instalado
            if not yesno("Não foi possível traduzir automaticamente. Deseja continuar sem traduzir?", default=False):
                print("Operação cancelada. Instale deep-translator e execute novamente.")
                return
            else:
                print("Continuando sem tradução.")
        else:
            text = translated
            detected_lang = target_lang
            print("✅ Texto traduzido e pronto para TTS.")

    # após tradução (ou não), escolher voz baseada no idioma final, mas pedir ao usuário qual voz quer
    short = detected_lang if detected_lang else None
    suggested = VOICE_LIST_BY_SHORT.get(short, None)
    print("\n🗣️ Escolha a VOZ para a leitura (baseada no idioma final):")
    menu = {}
    if suggested:
        for i, v in enumerate(suggested, start=1):
            menu[str(i)] = v
        manual_idx = len(suggested) + 1
    else:
        # mostrar padrão + exemplos
        examples = [DEFAULT_VOICE_PT, 'en-US-GuyNeural', 'en-US-JennyNeural']
        for i, v in enumerate(examples, start=1):
            menu[str(i)] = v
        manual_idx = len(examples) + 1
    menu[str(manual_idx)] = "Enter voice name manually"
    menu[str(manual_idx + 1)] = "Show more example voices"
    for k, v in menu.items():
        print(f" {k}) {v}")
    chosen_voice = None
    while True:
        choice = input("Número: ").strip()
        if not choice:
            print("Digite um número válido.")
            continue
        if choice in menu:
            sel = menu[choice]
            if sel == "Enter voice name manually":
                manual = input("✍️ Digite o nome exato da voz (ex: pt-BR-FranciscaNeural): ").strip()
                if manual:
                    chosen_voice = manual
                    break
                else:
                    print("Nome vazio.")
                    continue
            if sel == "Show more example voices":
                examples = [
                    'pt-BR-FranciscaNeural', 'pt-BR-HeloisaNeural', 'pt-BR-AntonioNeural',
                    'en-US-GuyNeural', 'en-US-JennyNeural', 'en-US-AriaNeural',
                    'es-ES-AlvaroNeural', 'es-ES-ElviraNeural'
                ]
                for i, ex in enumerate(examples, start=1):
                    print(f"  {i}) {ex}")
                s2 = input("Escolha número dos exemplos ou Enter para voltar: ").strip()
                if s2.isdigit():
                    idx = int(s2)
                    if 1 <= idx <= len(examples):
                        chosen_voice = examples[idx - 1]
                        break
                print("Voltando ao menu principal.")
                continue
            else:
                chosen_voice = sel
                break
        else:
            print("Opção inválida.")
    voice = chosen_voice
    print(f"\n🎤 Voice selecionada: {voice}\n")

    # pasta saída
    out_dir = cwd / (base_name or "output_audiobook")
    if out_dir.exists():
        if not yesno(f"Pasta '{out_dir.name}' existe. Usar ela?", default=True):
            i = 1
            while True:
                candidate = cwd / f"{(base_name or 'output_audiobook')}_{i}"
                if not candidate.exists():
                    out_dir = candidate
                    break
                i += 1
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"➡️ Usando pasta: {out_dir}\n")

    # max chars por parte TTS
    while True:
        lim_str = input(f"✂️ Max caracteres por parte (Enter para {MAX_CHARS}): ").strip()
        if not lim_str:
            limit = MAX_CHARS
            break
        try:
            limit = int(lim_str)
            if limit > 0:
                break
        except ValueError:
            pass
        print("Número inválido.")

    # split para TTS e gerar
    tts_parts = split_text_preserve_sentences(text, limit)
    total = len(tts_parts)
    print(f"📚 Texto dividido em {total} parte(s). Iniciando TTS com voice {voice}...\n")
    if total == 0:
        print("Nada para converter.")
        return
    if not yesno(f"Iniciar geração de {total} parte(s)?", default=True):
        print("Cancelado.")
        return

    prefix = "part"
    generated = []
    for i, p in enumerate(tts_parts, start=1):
        out_name = f"{prefix}{i:03d}.mp3"
        out_path = out_dir / out_name
        if out_path.exists():
            update_progress_line(i, total, f"⏭️ Pulando {out_name}")
            generated.append(str(out_path))
            await asyncio.sleep(0.05)
            continue
        update_progress_line(i, total, f"✳️ Gerando {out_name} (chars {len(p)})...")
        ok = await save_part_with_retries(p, str(out_path), voice)
        if not ok:
            print()
            print(f"\n❌ Falha ao gerar {out_name}. Abortando.")
            return
        update_progress_line(i, total, f"✅ Gerado {out_name}")
        generated.append(str(out_path))
        await asyncio.sleep(random.uniform(*DELAY_BETWEEN))
    print()

    ordered = sorted([Path(x).name for x in generated])
    list_path = out_dir / "list.txt"
    with list_path.open("w", encoding="utf-8") as f:
        for name in ordered:
            p = out_dir / name
            f.write(f"file '{p.as_posix()}'\n")
    print(f"\n➡️ Lista criada: {list_path}")

    output_mp3 = out_dir / "book_final.mp3"
    mp3_created = False
    if shutil.which("ffmpeg"):
        print("⚙️ Tentando juntar com ffmpeg...")
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(list_path), "-c", "copy", str(output_mp3)]
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if r.returncode == 0:
            mp3_created = True
            print(f"✅ MP3 final criado: {output_mp3}")
        else:
            print("⚠️ Modo rápido falhou. Tentando re-encode...")
            cmd2 = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(list_path), "-c:a", "libmp3lame", "-b:a", "192k", str(output_mp3)]
            r2 = subprocess.run(cmd2, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if r2.returncode == 0:
                mp3_created = True
                print(f"✅ MP3 criado (re-encoded): {output_mp3}")
            else:
                print("❌ Erro com ffmpeg.")

    if not mp3_created:
        try:
            with output_mp3.open("wb") as out:
                for name in ordered:
                    p = out_dir / name
                    print(f"   ➕ Adicionando {p.name}")
                    with p.open("rb") as part:
                        out.write(part.read())
            mp3_created = True
            print(f"✅ MP3 criado (binary): {output_mp3}")
        except Exception as e:
            print(f"❌ Falha ao criar MP3 final: {e}")

    if not mp3_created:
        print("❌ Não foi possível criar o MP3 final.")
        return

    # MP4 opcional
    if yesno("\n🎥 Deseja criar MP4 com o áudio?", default=False):
        if AudioFileClip is None:
            print("⚠️ MoviePy não instalado; instale moviepy para criar MP4.")
        else:
            mp4_output = input("Nome do vídeo (Enter para book_final.mp4): ").strip() or "book_final.mp4"
            mp4_path = out_dir / mp4_output
            do_cover = yesno("Usar imagem de capa?", default=False)
            cover_path = None
            if do_cover:
                images_found = [f for f in cwd.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_IMG_EXTS]
                images_found.sort(key=lambda f: f.name.lower())
                if images_found:
                    for i, img in enumerate(images_found, 1):
                        print(f" {i}) {img.name}")
                else:
                    print("Nenhuma imagem encontrada.")
                while True:
                    user_input = input("Escolha número, 0 para digitar caminho, q para pular: ").strip()
                    if user_input.lower() in ('q','quit','exit','skip'):
                        cover_path = None
                        break
                    if user_input == '0':
                        manual = input("Caminho da imagem: ").strip()
                        cand = Path(manual)
                        if cand.exists() and cand.is_file():
                            cover_path = cand
                            break
                        else:
                            print("Arquivo não encontrado.")
                            continue
                    if user_input.isdigit():
                        idx = int(user_input)
                        if 1 <= idx <= len(images_found):
                            cover_path = images_found[idx - 1]
                            break
                        else:
                            print("Número inválido.")
                            continue
                    cand_direct = Path(user_input)
                    if cand_direct.exists() and cand_direct.is_file():
                        cover_path = cand_direct
                        break
                    print("Opção inválida.")
            try:
                audio_clip = AudioFileClip(str(output_mp3))
                if cover_path:
                    video_clip = ImageClip(str(cover_path))
                else:
                    video_clip = ColorClip(size=(1280, 720), color=(0,0,0))
                video_clip = video_clip.set_duration(audio_clip.duration).set_audio(audio_clip)
                video_clip.write_videofile(str(mp4_path), fps=1, codec="libx264", audio_codec="aac", verbose=False, logger="bar")
                print(f"✅ MP4 criado: {mp4_path}")
            except Exception as e:
                print(f"❌ Erro criando MP4: {e}")

    print(f"\n✨ Pronto! Arquivos em: {out_dir.resolve()}\n")

if __name__ == '__main__':
    try:
        asyncio.run(generate_and_join())
    except KeyboardInterrupt:
        print("\nCancelado pelo usuário.")
