import os
import torch
import torchvision
from utils import set_random_seeds, create_model, prepare_dataloader, prepare_flowers102_dataloader, train_model, save_model, load_model, evaluate_model, create_classification_report


def main():

    random_seed = 0
    num_classes = 10
    # num_classes = 102
    l2_regularization_strength = 1e-4
    # learning_rate = 1e-1
    learning_rate = 1e-2
    # num_epochs = 200
    num_epochs = 10
    train_batch_size = 512
    # train_batch_size = 128
    cuda_device = torch.device("cuda:0")
    cpu_device = torch.device("cpu:0")

    model_dir = "saved_models"
    # model_filename = "resnet18_cifar10_test.pt"
    model_filename = "resnet50_cifar10_test.pt"
    model_filepath = os.path.join(model_dir, model_filename)

    set_random_seeds(random_seed=random_seed)

    # Create an untrained model.
    # model_func = torchvision.models.resnet18
    model_func = torchvision.models.resnet50
    model = create_model(num_classes=num_classes, model_func=model_func)

    # train_loader, test_loader, classes = prepare_dataloader(
    #     num_workers=8, train_batch_size=128, eval_batch_size=256)

    train_loader, test_loader, classes = prepare_dataloader(
        num_workers=8, train_batch_size=train_batch_size, eval_batch_size=512)

    # train_loader, test_loader, classes = prepare_flowers102_dataloader(
    #     num_workers=8, train_batch_size=train_batch_size, eval_batch_size=32
    # )

    # Train model.
    print("Training Model...")
    model = train_model(model=model,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        device=cuda_device,
                        l2_regularization_strength=l2_regularization_strength,
                        learning_rate=learning_rate,
                        num_epochs=num_epochs)

    # Save model.
    save_model(model=model, model_dir=model_dir, model_filename=model_filename)
    # Load a pretrained model.
    model = load_model(model=model,
                       model_filepath=model_filepath,
                       device=cuda_device)

    _, eval_accuracy = evaluate_model(model=model,
                                      test_loader=test_loader,
                                      device=cuda_device,
                                      criterion=None)

    classification_report = create_classification_report(
        model=model, test_loader=test_loader, device=cuda_device)

    print("Test Accuracy: {:.3f}".format(eval_accuracy))
    print("Classification Report:")
    print(classification_report)


if __name__ == "__main__":

    main()