"""Export a self-contained game replay HTML for local screen recording."""
import argparse
import html
import json
from pathlib import Path

import chess.pgn
import chess.svg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pgn", required=True)
    parser.add_argument("--game", type=int, default=1)
    parser.add_argument("--out", default="runs/replay.html")
    args = parser.parse_args()
    if args.game < 1:
        parser.error("game numbers start at 1")
    with open(args.pgn, encoding="utf-8") as source:
        for _ in range(args.game):
            game = chess.pgn.read_game(source)
            if game is None:
                raise ValueError("Requested game does not exist")
    frames = []
    board = game.board()
    frames.append({"svg": chess.svg.board(board, size=640), "move": "Starting position"})
    for index, move in enumerate(game.mainline_moves()):
        san = board.san(move)
        label = f"{board.fullmove_number}{'.' if board.turn else '...'} {san}"
        board.push(move)
        checked = board.king(board.turn) if board.is_check() else None
        frames.append({"svg": chess.svg.board(board, size=640, lastmove=move, check=checked), "move": label})
    title = f"{game.headers.get('White', 'White')} vs {game.headers.get('Black', 'Black')}"
    # Escape '<' in embedded JSON so a PGN label cannot terminate the script.
    payload = json.dumps(frames).replace("<", "\\u003c")
    page = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ScratchChess replay</title><style>
*{box-sizing:border-box}body{margin:0;background:#10171e;color:#ecf4f8;font:18px system-ui;display:grid;place-items:center;min-height:100vh}
main{width:min(94vw,660px);padding:24px 10px}h1{font-size:28px;margin:6px 0}.tag{color:#8bdac7;font-size:13px;letter-spacing:2px}
.sub{color:#b1c0cd;margin:8px 0 20px}#board svg{width:100%;height:auto;display:block}#move{text-align:center;font-size:26px;margin:16px}
.controls{display:flex;gap:10px;align-items:center;justify-content:center;flex-wrap:wrap}button,select{font:inherit;padding:8px 16px;border:1px solid #446073;background:#243747;color:white;border-radius:8px;cursor:pointer}
input{width:100%;margin:16px 0;accent-color:#8bdac7}small{color:#a4b4c0}.hide .controls,.hide input,.hide small{visibility:hidden}
</style><main><div class="tag">MY CHESS NEURAL NETWORK</div><h1>Learning to play.</h1><div class="sub">__TITLE__ · __RESULT__</div>
<div id="board"></div><div id="move"></div><input id="seek" type="range" min="0" value="0" aria-label="Game move">
<div class="controls"><button id="prev" aria-label="Previous move">←</button><button id="play">Play</button><button id="next" aria-label="Next move">→</button>
<select id="speed" aria-label="Playback speed"><option value="1000">1 sec / move</option><option value="500">0.5 sec / move</option><option value="2000">2 sec / move</option></select></div>
<p><small>Arrow keys: step · Space: play/pause · H: hide controls for recording. This is a saved game, not live analysis.</small></p></main>
<script>const frames=__FRAMES__;let index=0,timer=null;const board=document.getElementById('board'),label=document.getElementById('move'),seek=document.getElementById('seek'),play=document.getElementById('play');seek.max=frames.length-1;
function show(n){index=Math.max(0,Math.min(frames.length-1,n));board.innerHTML=frames[index].svg;label.textContent=frames[index].move;seek.value=index}
function stop(){clearInterval(timer);timer=null;play.textContent='Play'}
function toggle(){if(timer){stop();return}if(index===frames.length-1)show(0);play.textContent='Pause';timer=setInterval(()=>{show(index+1);if(index===frames.length-1)stop()},Number(document.getElementById('speed').value))}
play.onclick=toggle;document.getElementById('prev').onclick=()=>{stop();show(index-1)};document.getElementById('next').onclick=()=>{stop();show(index+1)};seek.oninput=()=>{stop();show(Number(seek.value))};document.getElementById('speed').onchange=()=>{if(timer){stop();toggle()}};
document.onkeydown=e=>{if(['INPUT','SELECT'].includes(e.target.tagName))return;if(e.key===' '){e.preventDefault();toggle()}if(e.key==='ArrowRight'){stop();show(index+1)}if(e.key==='ArrowLeft'){stop();show(index-1)}if(e.key.toLowerCase()==='h')document.body.classList.toggle('hide')};show(0);
</script></html>"""
    page = page.replace("__TITLE__", html.escape(title)).replace("__RESULT__", html.escape(game.headers.get("Result", "*"))).replace("__FRAMES__", payload)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
    print(f"Open {path.resolve()} in a browser. All board frames are embedded; no network connection needed.")


if __name__ == "__main__":
    main()
