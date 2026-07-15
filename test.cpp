#include <iostream>
#include <numeric>
#include <algorithm>
#include <cassert>
#include "numplusplus.hpp"


void testConstruction(){
    npp::array<int, 3> a({2, 3, 4});
    assert(a.rank() == 3);
    assert(a.size() == 24);

    auto s = a.shape();
    assert(s[0] == 2);
    assert(s[1] == 3);
    assert(s[2] == 4);

    std::cout << "[OK] construction\n";
}


void testFill(){
    npp::array<int, 2> a({10, 10});
    a.fill(7);
    for (const auto& x : a){
        assert(x == 7);
    }

    std::cout << "[OK] fill\n";
}


void testIteratorWrite(){
    npp::array<int, 1> a({100});

    int counter = 0;

    for (auto& x : a){
        x = counter++;
    }

    for (std::size_t i = 0; i < a.size(); ++i){
        assert(a(i) == static_cast<int>(i));
    }
    std::cout << "[OK] mutable iterator\n";
}


void testConstIterator(){
    npp::array<int, 1> a({50});

    int counter = 0;
    for (auto& x : a){
        x = counter++;
    }
    const auto& ca = a;

    int sum = std::accumulate(ca.cbegin(), ca.cend(),0);

    assert(sum == 1225); 

    std::cout << "[OK] const iterator\n";
}


void testStdAlgorithms(){
    npp::array<int, 1> a({5});

    int values[] = {5,2,8,1,9};

    std::copy(std::begin(values), std::end(values),a.begin());

    assert(*std::min_element(a.begin(), a.end()) == 1);
    assert(*std::max_element(a.begin(), a.end()) == 9);

    std::cout << "[OK] STL compatibility\n";
}


void testData(){
    npp::array<int, 2> a({3,3});

    int* ptr = a.data();
    assert(ptr != nullptr);

    ptr[0] = 123;
    assert(a(0,0) == 123);

    const auto& ca = a;
    const int* cptr = ca.data();
    assert(cptr[0] == 123);

    std::cout << "[OK] data()\n";
}


void testIndexing(){
    npp::array<int, 3> a({2,3,4});
    a(1,2,3) = 999;
    assert(a(1,2,3) == 999);

    std::cout << "[OK] indexing\n";
}


void testBounds(){
    npp::array<int, 2> a({3,4});
    bool thrown = false;

    try{
        a(3,0);
    }
    catch(const std::out_of_range&){
        thrown = true;
    }

    assert(thrown);

    thrown = false;
    try{
        a(0,4);
    }
    catch(const std::out_of_range&){
        thrown = true;
    }

    assert(thrown);

    std::cout << "[OK] bounds checking\n";
}


void testCopy(){
    npp::array<int,2> a({2,2});
    a.fill(42);
    auto b = a;
    assert(b.size() == a.size());

    for(std::size_t i = 0; i < a.size(); ++i){
        assert(a.data()[i] == b.data()[i]);
    }

    std::cout << "[OK] copy\n";
}


void testEmptyShapes(){
    npp::array<int,1> a({0});

    assert(a.size() == 0);
    assert(a.begin() == a.end());

    std::cout << "[OK] empty array\n";
}


int main(){
    testConstruction();
    testFill();
    testIteratorWrite();
    testConstIterator();
    testStdAlgorithms();
    testData();
    testIndexing();
    testBounds();
    testCopy();
    testEmptyShapes();

    std::cout << "\nALL TESTS PASSED\n";
}