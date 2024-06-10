import content_based, svd, DQN
def main():
    while(True):
        print("\t### MENU ###")
        print("1. Content based")
        print("2. SVD")
        print("3. DQN")
        print("4. Exit")
        algo = int(input("Please enter the algorithm you like: "))
        if(algo == 1): content_based.content_based()
        elif(algo == 2): svd.svd_main()
        elif(algo == 3): DQN.DQN()
        elif(algo == 4): break
        else: print("Invalid choice, please enter again.")
        print()

if __name__ == '__main__':
    main()
