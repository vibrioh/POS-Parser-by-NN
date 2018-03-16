#!/usr/bin/env bash

# part 1
python src/parser.py --vocabs model/vocabs --model model/model
python src/depModel.py trees/dev.conll outputs/dev_part1.conll
python src/eval.py trees/dev.conll outputs/dev_part1.conll
python src/depModel.py trees/test.conll outputs/test_part1.conll

# part 2
python src/parser.py --vocabs model/vocabs --model model/model --hidden1 400 --hidden2 400
python src/depModel.py trees/dev.conll outputs/dev_part2.conll
python src/eval.py trees/dev.conll outputs/dev_part2.conll
python src/depModel.py trees/test.conll outputs/test_part2.conll

# part 3
python src/parser.py --vocabs model/vocabs --model model/model --hidden1 600 --hidden2 400 --we 128 --pe 64 --le 64 --epochs 8 --minibatch 500
python src/depModel.py trees/dev.conll outputs/dev_part3.conll
python src/eval.py trees/dev.conll outputs/dev_part3.conll
python src/depModel.py trees/test.conll outputs/test_part3.conll