from brain.knowgraph_english import KnowledgeGraph


vocab_file = "D:\Downloads\ent_vocab_custom"
kg = KnowledgeGraph(vocab_file=vocab_file, predicate=True)
text = "Delhi is the capital of India ."
tokens, pos, vm, tag = kg.add_knowledge_with_vm([text], add_pad=False, max_length=16)
print(tag)
print(pos)
print(tokens)
print(vm)
