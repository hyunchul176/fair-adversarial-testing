"""Durable reservation ledgers; no simulator imports."""
from pathlib import Path
from src.evaluation.adversarial_a2_support import write


class NativeBudget:
    LIMITS={'sentinel':(16,384,1920),'initial_context':(48,1152,5760),'online_replan':(48,1152,5760)}
    def __init__(self,path=None):
        self.path=Path(path) if path else None
        self.rows={k:dict(starts=0,completed=0,actors=0,ticks=0,measured_ticks=0,uncertain_failed_calls=0,pending=False) for k in self.LIMITS}
    def start(self,phase):
        r=self.rows[phase]
        if r['starts']>=self.LIMITS[phase][0] or r['pending']:raise RuntimeError('Native reset budget exhausted/pending')
        r['starts']+=1;self.save()
    def reserve_action(self,phase):
        r=self.rows[phase];_,a,t=self.LIMITS[phase]
        if r['pending'] or r['actors']>=a or r['ticks']+5>t or sum(x['ticks'] for x in self.rows.values())+5>13440:
            raise RuntimeError('Native action budget exhausted/pending')
        r['actors']+=1;r['ticks']+=5;r['pending']=True;self.save()
    def charge(self,phase,measured,failed=False):
        r=self.rows[phase]
        if not r['pending'] or not 0<=measured<=5 or (not failed and measured<1):raise RuntimeError('Invalid native accounting')
        if not failed:r['ticks']+=measured-5
        r['measured_ticks']+=measured;r['uncertain_failed_calls']+=int(failed);r['pending']=False;self.save()
    def complete(self,phase):
        r=self.rows[phase]
        if r['pending'] or r['completed']>=r['starts']:raise RuntimeError('Invalid completion')
        r['completed']+=1;self.save()
    def save(self):
        if self.path:
            temp=self.path.with_suffix('.tmp');write(temp,self.rows);temp.replace(self.path)


class ModelBudget:
    LIMITS={'initial_context':(5489280,45744,48),'online_replan':(23564160,225984,768),'archived':(480000,7680,0)}
    EPISODE={'initial_context':(114360,953,1),'online_replan':(490920,4708,16)}
    def __init__(self,path=None):
        self.path=Path(path) if path else None
        self.rows={k:dict(model_ticks=0,candidate_forecasts=0,planning_calls=0) for k in self.LIMITS};self.episodes={}
    def reserve(self,phase,key,ticks,candidates):
        if not isinstance(ticks,int) or not isinstance(candidates,int) or ticks<1 or candidates<1:raise ValueError('Invalid model reservation')
        add=(ticks,candidates,int(phase!='archived'));names=('model_ticks','candidate_forecasts','planning_calls')
        row=self.rows[phase];ep=self.episodes.get(key,dict.fromkeys(names,0))
        if any(row[n]+a>cap for n,a,cap in zip(names,add,self.LIMITS[phase])):raise RuntimeError('Model phase budget exhausted')
        if phase in self.EPISODE and any(ep[n]+a>cap for n,a,cap in zip(names,add,self.EPISODE[phase])):raise RuntimeError('Model episode budget exhausted')
        for n,a in zip(names,add):row[n]+=a;ep[n]+=a
        self.episodes[key]=ep;self.save()
    def save(self):
        if self.path:
            temp=self.path.with_suffix('.tmp');write(temp,dict(phases=self.rows,episodes=self.episodes));temp.replace(self.path)
