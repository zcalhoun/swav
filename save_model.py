import torch
import src.resnet50 as resnet_models
import argparse

def main(args):
    # Load the architecture and the weights
    model = resnet_models.__dict__["resnet50"](output_dim=0, eval_mode=True)
    model_path = args.model_path
    state_dict = torch.load(model_path)

    # Pulled this code from eval_linear.py
    # This loads the relevant parts of the state dict on top of the model.

    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    # remove prefixe "module."
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    for k, v in model.state_dict().items():
        if k not in list(state_dict):
            print('key "{}" could not be found in provided state dict'.format(k))
        elif state_dict[k].shape != v.shape:
            print('key "{}" is of different shape in model and provided state dict'.format(k))
            state_dict[k] = v
    msg = model.load_state_dict(state_dict, strict=False)
    print("Load pretrained model with msg: {}".format(msg))

    # Save the model in the desired path for future use.
    torch.save(model.state_dict(), args.output_path)


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Saving resulting model as PyTorch file")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/path/to/last/checkpoint",
        help="path to last checkpoint" 
    )

    parser.add_argument(
        "--output_path",
        type=str,
        help='path to output the .pt model'
    )

    args = parser.parse_args()
    main(args)
    

