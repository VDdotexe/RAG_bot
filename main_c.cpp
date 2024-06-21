#include <iostream>
#include <fstream>
#include <string>

int main() {
    std::ifstream csvFile("data.csv");
    std::ofstream txtFile("data.txt");

    if (!csvFile.is_open() || !txtFile.is_open()) {
        std::cout << "Unable to open file\n";
        return 1;
    }

    std::string line;
    while (getline(csvFile, line)) {
        txtFile << line << "\n";
    }

    csvFile.close();
    txtFile.close();

    return 0;
}