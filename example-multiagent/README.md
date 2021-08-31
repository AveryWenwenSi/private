
[![Demo Macad Multiagent](https://share.gifyoutube.com/oVDLMk.gif)](https://www.youtube.com/watch?v=jQ4GX_W1MZE)


Integrate the original arthur's multi-agent code

Just simply run

```python
python basic_agent.py
```

Ray Result at ~/ray_results
open website at http://localhost:8265 (127.0.0.1:8265) to see ray dashboard

**Change config here: https://github.com/eerkaijun/macad-gym/blob/master/example-multiagent/basic_agent.py#L128-L153**

no additional requirements
(maybe ray==0.6.2)

Have tested that in macad-gym you can install without conflict:

torch==1.4.0
tensorflow-gpu==1.15


For faster graphics you can change the workers-num(cpu core)
and batch_size and step_size

Also, notice the resolution (observation space) should match from the network,

change that at https://github.com/praveen-palanisamy/macad-gym/blob/38884ac4bc7fb2e91a865950cff4eeadeed02f60/src/macad_gym/envs/homo/ncom/inde/po/intrx/ma/stop_sign_3c_town03.py#L17-L18

with

```python
            "x_res": 84,
            "y_res": 84,

```
you should chage both the env and the 3 actors resolution! 
