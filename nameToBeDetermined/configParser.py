from ruamel.yaml import YAML
yaml = YAML(typ='safe')
with open('config.yaml') as file:
    config = yaml.load(file)


embeddingDim = config['pretrain']['embeddingDim']
