import argparse
import os
import time
import torch
import torchvision
from utils import (set_random_seeds, create_model, prepare_cifar10_dataloader,
                   train_model, save_model, load_model, evaluate_model,
                   create_classification_report)


def main():

    parser = argparse.ArgumentParser(
        description="Train ResNet50 on CIFAR-10",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--use_amp",
                        action="store_true",
                        help="Use automatic mixed precision training.")
    parser.add_argument("--model_dir",
                        type=str,
                        default="saved_models",
                        help="Directory to save model checkpoints.")
    parser.add_argument("--model_filename",
                        type=str,
                        required=True,
                        help="Model filename to save.")

    args = parser.parse_args()

    random_seed = 0
    l2_regularization_strength = 5e-4
    learning_rate = 1e-1
    num_epochs = 200
    train_batch_size = 512
    eval_batch_size = 512
    cuda_device = torch.device("cuda:0")

    use_amp = args.use_amp
    model_dir = args.model_dir
    model_filename = args.model_filename
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)

    set_random_seeds(random_seed=random_seed)

    # Prepare dataset.
    train_loader, test_loader, class_names = prepare_cifar10_dataloader(
        num_workers=8,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size)

    # Create an untrained model.
    # Use ResNet50 to demonstrate the acceleration of mixed precision training.
    # ResNet18 shows almost no significant speed-up.
    model_func = torchvision.models.resnet50
    num_classes = len(class_names)
    model = create_model(num_classes=num_classes, model_func=model_func)

    # Train model.
    print("Training Model...")
    # Record the start time.
    start_time = time.time()
    model = train_model(model=model,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        device=cuda_device,
                        l2_regularization_strength=l2_regularization_strength,
                        learning_rate=learning_rate,
                        num_epochs=num_epochs,
                        use_amp=use_amp)
    # Record the end time.
    end_time = time.time()
    # Convert the elapsed time to HH:MM:SS format.
    elapsed_time = time.strftime("%H:%M:%S",
                                 time.gmtime(end_time - start_time))
    print(f"Training Elapsed Time: {elapsed_time}")

    # Save model.
    save_model(model=model, model_dir=model_dir, model_filename=model_filename)
    # Load a pretrained model.
    model = load_model(model=model,
                       model_filepath=model_filepath,
                       device=cuda_device)

    # Evaluate model.
    print("Evaluating Model...")
    _, eval_accuracy = evaluate_model(model=model,
                                      test_loader=test_loader,
                                      device=cuda_device,
                                      criterion=None)

    classification_report = create_classification_report(
        model=model, test_loader=test_loader, device=cuda_device)

    print(f"Test Accuracy: {eval_accuracy:.3f}")
    print("Classification Report:")
    print(classification_report)


if __name__ == "__main__":

    main()
