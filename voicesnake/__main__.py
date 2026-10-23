import argparse
import csv
from pathlib import Path
import queue
import time


def main():
    p=argparse.ArgumentParser(description='Offline CPU four-class GRU voice-controlled Snake')
    p.add_argument('--keyboard-only',action='store_true')
    p.add_argument('--list-devices',action='store_true')
    p.add_argument('--doctor',action='store_true',help='Check dependencies, bundled model and input devices')
    p.add_argument('--device',type=int)
    p.add_argument('--threshold',type=float,default=.55)
    p.add_argument('--energy',type=float,default=.004)
    p.add_argument('--confirmations',type=int,default=2)
    p.add_argument('--interval',type=float,default=.08)
    p.add_argument('--speed',type=int,default=5)
    p.add_argument('--metrics',type=Path,help='CSV session log; defaults to the user data directory')
    p.add_argument('--smoke-frames',type=int,default=0,help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.doctor:
        from .doctor import main as doctor_main
        doctor_main()
        return
    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices()); return
    if not (0<args.threshold<=1 and args.energy>0 and args.confirmations>=1
            and .05<=args.interval<=1 and 2<=args.speed<=15):
        p.error('Invalid threshold, energy, confirmations, interval or speed')
    from .paths import data_home
    from datetime import datetime
    if args.metrics is None:
        args.metrics=data_home()/'sessions'/f"session-{datetime.now():%Y%m%d-%H%M%S-%f}.csv"
    import pygame
    from .game import Snake
    from .renderer import Renderer
    from .audio import AudioWorker, Status
    from .settings import Settings
    try:
        renderer=Renderer()
    except pygame.error as exc:
        p.exit(1,f'Cannot open the game window: {exc}. Run on a desktop with a display.\n')
    game=Snake()
    settings=Settings(args.threshold,args.energy,args.confirmations)
    settings_open=False
    auto_start=not args.keyboard_only
    game.paused=auto_start
    worker=None if args.keyboard_only else AudioWorker(args.threshold,args.confirmations,args.energy,args.interval,args.device,settings)
    voice=not args.keyboard_only
    if worker: worker.start()
    args.metrics.parent.mkdir(parents=True,exist_ok=True)
    file=args.metrics.open('w',newline='')
    log=csv.writer(file)
    log.writerow(['event','direction','confidence','detected_at','audio_onset_proxy','applied_at','detect_to_turn_ms','onset_proxy_to_detect_ms'])
    clock=pygame.time.Clock(); running=True; speed=args.speed
    deadline=time.perf_counter()+1/speed
    last='-'; turn_ms=0.; pending=None; frames=0
    keymap={pygame.K_UP:'up',pygame.K_w:'up',pygame.K_DOWN:'down',pygame.K_s:'down',
            pygame.K_LEFT:'left',pygame.K_a:'left',pygame.K_RIGHT:'right',pygame.K_d:'right'}
    try:
        while running:
            now=time.perf_counter()
            for event in pygame.event.get():
                if event.type==pygame.QUIT: running=False
                if settings_open and event.type==pygame.MOUSEBUTTONDOWN and event.button==1:
                    for name,control in list(renderer.controls.items()):
                        if len(control)!=5: continue
                        minus,plus,low,high,step=control
                        if minus.collidepoint(renderer.to_canvas(event.pos)) or plus.collidepoint(renderer.to_canvas(event.pos)):
                            delta=step if plus.collidepoint(renderer.to_canvas(event.pos)) else -step
                            if name=='threshold': settings.update_voice(threshold=settings.voice().threshold+delta)
                            elif name=='energy': settings.update_voice(energy=settings.voice().energy+delta)
                            elif name=='confirmations': settings.update_voice(confirmations=settings.voice().confirmations+int(delta))
                            elif name=='speed': speed=max(2,min(15,speed+int(delta)))
                            break
                    continue
                if not settings_open and event.type==pygame.MOUSEBUTTONDOWN and event.button==1:
                    toggle=renderer.controls.get('toggle')
                    if toggle and toggle[0].collidepoint(renderer.to_canvas(event.pos)): settings_open=True
                    continue
                if event.type!=pygame.KEYDOWN: continue
                if event.key==pygame.K_TAB:
                    settings_open=not settings_open
                    continue
                if settings_open and event.key==pygame.K_ESCAPE:
                    settings_open=False
                    continue
                key=event.key
                if key==pygame.K_ESCAPE: running=False
                elif key==pygame.K_r:
                    game.reset(); pending=None; auto_start=False; deadline=now+1/speed
                elif key==pygame.K_SPACE:
                    game.paused=not game.paused; pending=None; auto_start=False; deadline=now+1/speed
                    game.pending_direction=game.direction
                elif key==pygame.K_v and worker:
                    voice=not voice; pending=None
                    game.pending_direction=game.direction
                elif key in (pygame.K_EQUALS,pygame.K_PLUS,pygame.K_KP_PLUS): speed=min(15,speed+1)
                elif key in (pygame.K_MINUS,pygame.K_KP_MINUS): speed=max(2,speed-1)
                elif key in keymap and game.turn(keymap[key]): pending=None
            if worker:
                if voice and not game.paused and not game.over: worker.set_enabled(True)
                else: worker.set_enabled(False)
                while True:
                    try: command=worker.commands.get_nowait()
                    except queue.Empty: break
                    if not voice or game.over or game.paused or now-command.detected_at>.25:
                        continue
                    last=f'{command.direction.upper()} {command.confidence:.0%}'
                    accepted=game.turn(command.direction)
                    if not accepted: last += ' BLOCKED: reverse'
                    elif command.direction==game.direction: last += ' ALREADY'
                    else: last += ' QUEUED'
                    onset_ms=(command.detected_at-command.onset_at)*1000 if command.onset_at is not None else ''
                    log.writerow(['accepted' if accepted else 'rejected',command.direction,command.confidence,
                                  command.detected_at,command.onset_at,'','',onset_ms])
                    if accepted: pending=command
            if now>=deadline:
                changed=game.tick()
                if pending:
                    applied=time.perf_counter()
                    if changed:
                        last=f"{pending.direction.upper()} APPLIED"
                        turn_ms=(applied-pending.detected_at)*1000
                        log.writerow(['turn',pending.direction,pending.confidence,pending.detected_at,
                                      pending.onset_at,applied,turn_ms,''])
                        file.flush()
                    pending=None
                deadline=now+1/speed
            status=worker.status if worker else Status(microphone='Keyboard mode')
            if auto_start and status.microphone=='Listening':
                auto_start=False; game.paused=False; deadline=now+1/speed
            renderer.draw(game,status,voice,speed,last,turn_ms,settings.voice(),settings_open)
            clock.tick(60); frames+=1
            if args.smoke_frames and frames>=args.smoke_frames: running=False
    finally:
        if worker: worker.close()
        file.close(); pygame.quit()

if __name__=='__main__': main()
