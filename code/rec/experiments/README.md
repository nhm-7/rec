# Uso de YAER para versionar experimentos

## Descripción y usos
La mayoría de científicos siempre tratan de hacer de sus experimentos reproducibles bajo ciertas condiciones iniciales, ó bien que se produzcan iguales salidas para las mismas entradas. En este trabajo no solo hicimos experimentos usando y modificando el modelo baseline descrito en el capítulo 2, sino que se hizo uso de una librería pública disponible en Github llamada Yet Another Experiment Runner (YAER).

Técnicamente, YAER es un conjunto de decoradores de Python y se basa en la idea de que un experimento es solo una función con argumentos. Lo que nos permite hacer esta librería es centralizar todos los experimentos en un solo archivo, de manera que la mayoría de los argumentos que utiliza un determinado experimento queden explícitamente definidos en un archivo de Python. De esta forma, la reproducibilidad de un mismo experimento queda relacionada estrictamente a la versión de código del mismo.

## Componentes
YAER posee dos componentes principales, que son funciones de alto nivel, a saber:
`experiment_component` y `experiment`. Estos decoradores permiten etiquetar a las funciones relacionadas a nuestros experimentos propios. Si un experimento es un conjunto de componentes que tienen alguna relación entre sí, dichos componentes serán etiquetados con el decorador `experiment_component`. Por otro lado, el experimento definido y que centraliza toda la configuración, lo etiquetamos con el decorador `experiment`, que establece un diccionario cuyas keys son los argumentos de un experimento, y los valores, son simplemente los valores que reciben esos argumentos al requerir la ejecución de algún experimento. Se puede consultar el código que define a ambos componentes directamente desde el [repositorio de YAER](https://github.com/arielrossanigo/yaer/tree/master).

## Cómo lo usamos en este trabajo
En este trabajo todos los experimentos fueron definidos en el archivo `exps.py`. Por ejemplo, el experimento exp_001 fué definido así:

```python
@experiment({
    "model_args": {
        "backbone": "resnet50",
        "mask_pooling": False,
        "dropout_p": 0.1,
        "num_heads": 8,
        "num_layers": 6,
        "num_conv": 6,
        "use_visual_embeddings": True,
        "use_visual_pos_embeddings": True,
        "visual_pos_emb": {
            "name": "learned_pos_emb_2d",
            "args": {
                "embedding_dim": 256,
            },
        },
    },
    "data_args": {
        "dataset": "refclef",
        "max_length": 32,
        "input_size": 512,
    },
    "loss_args": {
        "beta": 0.1,
        "gamma": 0.1,
        "mu": 0.1,
    },
    "trainer_args": {
        "learning_rate": 1e-4,
        "weight_decay": 0.0,
        "batch_size": 4,
        "grad_steps": 1,
        "max_epochs": 1,
        "scheduler": lambda _: {},
    },
    "runtime_args": {
        "gpus": None,
        "num_workers": 8,
        "seed": 3407,
        "suffix": None,
        "cache": "./cache",
        "debug": False,
        "early_stopping": False,
        "amp": False,
        "force_ddp": True,
        "profile": False,
        "checkpoint": None,
        "save_last": False,
        "pdata": 0.34,
        "output_dir": "exp_001",
        "get_sample": True
    }
})
def exp_001():
    """An experiment for testing purposes (refactors, etc)."""
    run_experiment(model_factory=lit_model_factory)
```

Como vemos, el decorador `experiment` define todos los argumentos que `exp_001` utilizará en tiempo de ejecución. Dichos conjuntos de argumentos se van a utilizar principalmente en `base.py`, pero muchos de ellos se actualizan en cascada en otros archivos de más "bajo nivel", como lo es `models.py` con la función `lit_model_factory`. En todos estos archivos, se utiliza el decorador `experiment_component`, dandole saber a YAER que en ese contexto hay que actualizar los argumentos durante la ejecución del experimento.