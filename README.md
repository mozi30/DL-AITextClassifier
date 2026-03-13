# AITextClassifier
TODO: Write description and setup

## Used Datasets

### OTB 

https://huggingface.co/datasets/MLNTeam-Unical/OpenTuringBench

### HC3


### GSINGH1

https://huggingface.co/datasets/gsingh1-py/train

## Current Data

### Entries: (records.json)

- chatgpt:  86279
- mistral:  52650
- gemma:  49627 (thats a version of gemini)
- llama:  50800
- human:  139676

### From Sources:

- HC3 samples: 197552
- GSINGH1 samples: 60161
- OTB samples: 121319
- Total samples: 379032

### Scheme (scheme.json)

```json
{
    "model": "chatgpt,human,mistral,llama,gemma", 
    "text": "afioweogwHO",
    "topic": "chemistry, physics,...",
    "length": 120,
    "origin": "hc3 or gsingh1"
}
```
