import pygame

BG=(13,19,30)
PANEL=(22,31,46)
TEXT=(229,237,247)
MUTED=(145,163,186)
GREEN=(96,229,169)


class Renderer:
    def __init__(self):
        pygame.init()
        desktop=pygame.display.get_desktop_sizes()[0]
        scale=min(1.,desktop[0]*.9/1080,desktop[1]*.9/860)
        self.window=pygame.display.set_mode((max(320,int(1080*scale)),max(240,int(860*scale))),pygame.RESIZABLE)
        self.screen=pygame.Surface((1080,860))
        pygame.display.set_caption('VoiceSnake | GRU - 4 Commands')
        self.font=pygame.font.SysFont('monospace',18)
        self.small=pygame.font.SysFont('monospace',14)
        self.title=pygame.font.SysFont('monospace',34,bold=True)
        self.settings_rect=pygame.Rect(608,570,448,240)
        self.controls={}

    def viewport(self):
        width,height=self.window.get_size()
        scale=min(width/1080,height/860)
        size=(max(1,int(1080*scale)),max(1,int(860*scale)))
        offset=((width-size[0])//2,(height-size[1])//2)
        return scale,size,offset

    def to_canvas(self, position):
        scale,_,offset=self.viewport()
        return ((position[0]-offset[0])/scale,(position[1]-offset[1])/scale)

    def text(self, value, x, y, color=TEXT, font=None):
        self.screen.blit((font or self.font).render(str(value),True,color),(x,y))

    def draw(self, game, status, enabled, speed, last, turn_ms, settings, settings_open):
        s=self.screen
        s.fill(BG)
        self.text('VoiceSnake',26,20,GREEN,self.title)
        self.text('Say UP / DOWN / LEFT / RIGHT',28,65,MUTED)
        origin=(28,112); cell=23
        pygame.draw.rect(s,PANEL,(24,108,560,560),border_radius=12)
        for x in range(game.size):
            for y in range(game.size):
                pygame.draw.rect(s,(27,39,55),(origin[0]+x*cell,origin[1]+y*cell,22,22),border_radius=3)
        if game.food:
            x,y=game.food
            pygame.draw.circle(s,(255,120,125),(origin[0]+x*cell+11,origin[1]+y*cell+11),7)
        for i,(x,y) in enumerate(game.body):
            pygame.draw.rect(s,GREEN if i==0 else (48,159,123),
                             (origin[0]+x*cell+1,origin[1]+y*cell+1,21,21),border_radius=5)
        self.text('SCORE',620,118,MUTED)
        self.text(f'{game.score:03d}',620,145,GREEN,self.title)
        rows=[('VOICE CONTROL','ON' if enabled else 'OFF'),('DIRECTION',game.direction.upper()),
              ('LAST COMMAND',last),('PREDICTION',f'{status.label}  {status.confidence:.1%}'),
              ('INFERENCE',f'{status.inference_ms:.1f} ms'),('DETECT TO TURN',f'{turn_ms:.1f} ms'),
              ('MIC RMS',f'{status.rms:.4f}'),('SPEED',f'{speed} cells / second'), ('DETECTOR',status.decision)]
        for i,(key,value) in enumerate(rows):
            y=205+i*33
            self.text(key,620,y,MUTED,self.small)
            self.text(value,620,y+15,font=self.small)
        self.text('MICROPHONE',620,510,MUTED,self.small)
        words=status.microphone.split(); line=''; lines=[]
        for word in words:
            if len(line)+len(word)>43:
                lines.append(line); line=''
            line+=word+' '
        lines.append(line)
        for i,line in enumerate(lines[:2]):
            self.text(line,620,528+i*17,MUTED,self.small)

        panel=pygame.Rect(608,570,448,112)
        if settings_open:
            pygame.draw.rect(s,PANEL,self.settings_rect,border_radius=12)
            self.text('VOICE SETTINGS',624,585,GREEN,self.font)
            values=(('threshold',settings.threshold,.1,1.,.01),
                    ('energy',settings.energy,.0005,.03,.0005),
                    ('confirmations',settings.confirmations,1,4,1),
                    ('speed',speed,2,15,1))
            self.controls={}
            for i,(name,value,low,high,step) in enumerate(values):
                y=620+i*40
                minus=pygame.Rect(790,y,34,30); plus=pygame.Rect(966,y,34,30)
                self.controls[name]=(minus,plus,low,high,step)
                self.text(name.upper(),624,y+6,MUTED,self.small)
                pygame.draw.rect(s,(39,55,74),minus,border_radius=5)
                pygame.draw.rect(s,(39,55,74),plus,border_radius=5)
                self.text('-',802,y+5,GREEN)
                self.text('+',978,y+5,GREEN)
                shown=f'{value:.3f}' if name=='energy' else (f'{value:.2f}' if name=='threshold' else value)
                self.text(shown,836,y+6)
        else:
            self.controls={}
            pygame.draw.rect(s,PANEL,panel,border_radius=9)
            self.text(f'VOICE  {settings.threshold:.2f}   ENERGY  {settings.energy:.3f}',620,585,GREEN,self.small)
            self.text('CONFIRM  '+str(settings.confirmations)+'     CLICK SETTINGS TO ADJUST',620,612,MUTED,self.small)
            button=pygame.Rect(865,642,138,29)
            pygame.draw.rect(s,(39,55,74),button,border_radius=6)
            self.text('SETTINGS',886,648,GREEN,self.small)
            self.controls['toggle']=(button,)
        self.text('Arrows/WASD: turn   SPACE: pause   R: restart',28,830,MUTED,self.small)
        self.text('V: voice   +/-: speed   Esc: quit',620,830,MUTED,self.small)
        if game.over or game.paused:
            overlay=pygame.Surface((552,552),pygame.SRCALPHA)
            overlay.fill((7,12,20,205)); s.blit(overlay,origin)
            self.text('GAME OVER' if game.over else 'PAUSED',145,330,GREEN,self.title)
            self.text('R to restart' if game.over else 'SPACE to resume',165,380)
        _,size,offset=self.viewport()
        self.window.fill(BG)
        self.window.blit(pygame.transform.smoothscale(self.screen,size),offset)
        pygame.display.flip()
