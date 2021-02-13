from src.entity_linker_model import EntityLinker

model = EntityLinker()

input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)