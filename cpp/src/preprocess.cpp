#include <iostream>
#include "common.hpp"

using namespace std;

std::size_t numbeOfFilesInDirectory(std::experimental::filesystem::path path)
{
    using std::experimental::filesystem::directory_iterator;
    using fp = bool (*)( const std::experimental::filesystem::path&);
    return std::count_if(directory_iterator(path), directory_iterator{},
                         (fp)std::experimental::filesystem::is_regular_file);
}

void processImagesDir(std::string folder)
{
    size_t n = numbeOfFilesInDirectory(folder);
    int r = 2;

    std::vector<std::string> fileList;
    for (auto & p : fs::directory_iterator(folder)) {
        std::string name = p.path();
        fileList.push_back(name);
    }

    std::vector<pair<int, int>> pairIds;
    std::vector<bool> v(n);
    std::fill(v.begin(), v.begin() + r, true);

    do {
       std::vector<int> pair;
       for (int i = 0; i < n; ++i) {
           if (v[i]) {
               pair.push_back(i);
           }
       }
       pairIds.push_back(std::make_pair(pair.at(0), pair.at(1)));
    } while (std::prev_permutation(v.begin(), v.end()));

    ofstream file;

    string resFile = "results.csv";
    file.open(resFile);
    for (auto& t : pairIds) {
        std::string left = fileList.at(t.first);
        std::string right = fileList.at(t.second);
        std::vector<float> res = getDistances68(left, right);
        //--------------------------------------MSE (begin)
        float error = 0;
        for (int i = 0; i < res.size(); ++i) {
            float dist = res.at(i);
            error += pow(dist, 2);
        }
        error = error / res.size();
        cout << left << " " << right << " mse: " << error << endl;
        //--------------------------------------MSE (end)
        for (int i = 0; i < res.size(); i++)
        {
            file << res[i] << ",";
        }

        if (folder.find("fake") != std::string::npos) {
            file << "1\n";
        } else {
            file << "0\n";
        }
    }
    file.close();
}

int main(int argc, char** argv)
{
    if ( argc != 2 )
    {
        printf("usage1: imageDir\n");
        return -1;
    }

    std::string dir = argv[1];
    processImagesDir(dir);

    return 0;
}
