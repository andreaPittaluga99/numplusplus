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

void testDouble(){

    npp::array<double,2> a({2,2},3.14);

    for(auto x:a){
        assert(x==3.14);
    }
    std::cout << "[OK] double\n";
}

void testValueConstructor(){
    npp::array<int, 2> a({3,4}, 42);

    for (const auto& x : a){
        assert(x == 42);
    }

    std::cout << "[OK] value constructor\n";
}


void testNegativeIndexing(){
    npp::array<int, 2> a({2,3});

    int value = 0;
    for (auto& x : a){
        x = value++;
    }

    assert(a(-1,-1) == 5);
    assert(a(-1,0) == 3);
    assert(a(0,-1) == 2);

    std::cout << "[OK] negative indexing\n";
}


void testNegativeIndexOutOfBounds(){
    npp::array<int, 2> a({2,3});
    bool thrown = false;

    try{
        a(-3,0);
    }
    catch(const std::out_of_range&){
        thrown = true;
    }

    assert(thrown);

    thrown = false;

    try{
        a(0,-4);
    }
    catch(const std::out_of_range&){
        thrown = true;
    }
    assert(thrown);

    std::cout << "[OK] negative bounds checking\n";
}


void testStride(){
    npp::array<int,3> a({2,3,4});

    auto s = a.stride();

    assert(s[0] == 12);
    assert(s[1] == 4);
    assert(s[2] == 1);

    std::cout << "[OK] stride\n";
}


void testStrideAfterReshape(){
    npp::array<int,3> a({2,3,4});

    a.reshape({4,2,3});
    auto s = a.stride();

    assert(s[0] == 6);
    assert(s[1] == 3);
    assert(s[2] == 1);

    std::cout << "[OK] stride after reshape\n";
}


void testZeroDimension2d(){
    npp::array<int,2> a({0,5});

    assert(a.size() == 0);
    assert(a.begin() == a.end());
    assert(a.data()==nullptr);
    assert(a.shape()[0] == 0);
    assert(a.shape()[1] == 5);

    std::cout << "[OK] zero dimension 2d\n";
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


void testReshapeBasic(){
    npp::array<int, 2> a({3, 4});
    int value = 0;
    for (auto& x : a){
        x = value++;
    }

    assert(a.shape()[0] == 3);
    assert(a.shape()[1] == 4);

    a.reshape({2, 6});

    assert(a.shape()[0] == 2);
    assert(a.shape()[1] == 6);

    for (std::size_t i = 0; i < 12; ++i)
        assert(a.data()[i] == static_cast<int>(i));

    std::cout << "[OK] reshape basic\n";
}


void testReshapeIndexing(){
    npp::array<int, 2> a({2, 3});

    int value = 0;
    for (auto& x : a){
        x = value++;
    }

    /*  Before:

        0 1 2
        3 4 5

        after:

        0 1
        2 3
        4 5     */

    a.reshape({3, 2});

    assert(a(0,0) == 0);
    assert(a(0,1) == 1);

    assert(a(1,0) == 2);
    assert(a(1,1) == 3);

    assert(a(2,0) == 4);
    assert(a(2,1) == 5);

    std::cout << "[OK] resshape indexing\n";
}


void testReshapeInvalid(){
    npp::array<int, 2> a({3, 4});

    bool thrown = false;

    try{
        a.reshape({5, 5});
    }
    catch (const std::invalid_argument&){
        thrown = true;
    }

    assert(thrown);

    std::cout << "[OK] reshape invalid\n";
}


void testReshape3d(){
    npp::array<int, 3> a({2, 3, 4});

    int value = 0;
    for (auto& x : a){
        x = value++;
    }

    a.reshape({4, 3, 2});

    assert(a.shape()[0] == 4);
    assert(a.shape()[1] == 3);
    assert(a.shape()[2] == 2);

    //last elem
    assert(a(3,2,1) == 23);

    std::cout << "[OK] reshape 3d\n";
}


void testCopyConstructor(){
    npp::array<int,2> a({2,3}, 7);

    npp::array<int,2> b = a;

    assert(b.shape()[0] == 2);
    assert(b.shape()[1] == 3);
    assert(b.size() == 6);

    for (auto x : b){
        assert(x == 7);
    }

    // verifica che siano indipendenti
    b(0,0) = 100;

    assert(a(0,0) == 7);
    assert(b(0,0) == 100);

    std::cout << "[OK] copy constructor\n";
}


void testCopyAssignment(){
    npp::array<int,2> a({2,3}, 5);
    npp::array<int,2> b({4,4}, 9);

    b = a;

    assert(b.shape()[0] == 2);
    assert(b.shape()[1] == 3);
    assert(b.size() == 6);

    for (auto x : b){
        assert(x == 5);
    }

    b(0,0) = 50;

    assert(a(0,0) == 5);

    std::cout << "[OK] copy assignment\n";
}


void testMoveConstructor(){
    npp::array<int,2> a({3,3}, 8);

    npp::array<int,2> b = std::move(a);

    assert(a.size()==0);
    assert(b.shape()[0] == 3);
    assert(b.shape()[1] == 3);
    assert(b.size() == 9);

    for (auto x : b){
        assert(x == 8);
    }

    std::cout << "[OK] move constructor\n";
}


void testMoveAssignment(){
    npp::array<int,2> a({2,4}, 6);
    npp::array<int,2> b({10,10}, 1);

    b = std::move(a);

    assert(b.shape()[0] == 2);
    assert(b.shape()[1] == 4);
    assert(b.size() == 8);

    for (auto x : b){
        assert(x == 6);
    }

    std::cout << "[OK] move assignment\n";
}

void testSwap(){
    npp::array<int,2> a({2,3});
    npp::array<int,2> b({3,2});

    int value = 0;
    for(auto& x : a){
        x = value++;
    }

    value = 100;
    for(auto& x : b){
        x = value++;
    }

    a.swap(b);

    // shape changed
    assert(a.shape()[0] == 3);
    assert(a.shape()[1] == 2);

    assert(b.shape()[0] == 2);
    assert(b.shape()[1] == 3);

    assert(a.size() == 6);
    assert(b.size() == 6);

    // storage changed
    assert(a(0,0) == 100);
    assert(a(0,1) == 101);
    assert(a(1,0) == 102);
    assert(a(1,1) == 103);
    assert(a(2,0) == 104);
    assert(a(2,1) == 105);

    assert(b(0,0) == 0);
    assert(b(0,1) == 1);
    assert(b(0,2) == 2);
    assert(b(1,0) == 3);
    assert(b(1,1) == 4);
    assert(b(1,2) == 5);

    // swap w self
    a.swap(a);

    assert(a.shape()[0] == 3);
    assert(a.shape()[1] == 2);
    assert(a(0,0) == 100);
    assert(a(2,1) == 105);


    std::cout << "[OK] swap\n";
}


void testArrayArithmetic(){
    npp::array<int,1> a({5});
    npp::array<int,1> b({5});

    for(std::size_t i = 0; i < 5; ++i){
        a(i) = static_cast<int>(i);
        b(i) = 10;
    }

    auto c = a + b;

    assert(c(0) == 10);
    assert(c(1) == 11);
    assert(c(2) == 12);
    assert(c(3) == 13);
    assert(c(4) == 14);

    auto d = b - a;

    assert(d(0) == 10);
    assert(d(1) == 9);
    assert(d(2) == 8);
    assert(d(3) == 7);
    assert(d(4) == 6);

    auto e = a * b;

    assert(e(0) == 0);
    assert(e(1) == 10);
    assert(e(2) == 20);
    assert(e(3) == 30);
    assert(e(4) == 40);

    auto f = b / b;

    for(auto x : f){
        assert(x == 1);
    }

    std::cout << "[OK] array arithmetic\n";
}


void testCompoundArithmetic(){
    npp::array<int,1> a({5}, 2);
    npp::array<int,1> b({5}, 3);

    a += b;
    for(auto x : a){
        assert(x == 5);
    }

    a -= b;
    for(auto x : a){
        assert(x == 2);
    }


    a *= b;
    for(auto x : a){
        assert(x == 6);
    }

    a /= b;
    for(auto x : a){
        assert(x == 2);
    }

    std::cout << "[OK] compound arithmetic\n";
}


void testScalarArithmetic(){
    npp::array<int,1> a({5}, 10);

    auto b = a + 5;
    for(auto x : b){
        assert(x == 15);
    }

    auto c = a - 5;
    for(auto x : c){
        assert(x == 5);
    }

    auto d = a * 2;
    for(auto x : d){
        assert(x == 20);
    }

    auto e = a / 2;
    for(auto x : e){
        assert(x == 5);
    }

    std::cout << "[OK] scalar arithmetic\n";
}


void testReverseScalarArithmetic(){
    npp::array<int,1> a({5}, 10);

    auto b = 5 + a;

    for(auto x : b){
        assert(x == 15);
    }


    auto c = 20 - a;

    for(auto x : c){
        assert(x == 10);
    }


    auto d = 2 * a;

    for(auto x : d){
        assert(x == 20);
    }


    auto e = 100 / a;

    for(auto x : e){
        assert(x == 10);
    }

    std::cout << "[OK] reverse scalar arithmetic\n";
}

void testSizeOverflow(){
    bool thrown = false;

    try{
        npp::array<int,2> a({std::numeric_limits<std::size_t>::max(),2});
    }
    catch(const std::overflow_error&){
        thrown = true;
    }

    assert(thrown);

    std::cout << "[OK] size overflow\n";
}

void testArithmeticShapeMismatch(){
    npp::array<int,1> a({3}, 1);
    npp::array<int,1> b({4}, 1);

    bool thrown = false;

    try{
        auto c = a + b;
    }
    catch(const std::invalid_argument&){
        thrown = true;
    }

    assert(thrown);

    std::cout << "[OK] arithmetic shape mismatch\n";
}

void testEquality(){

    npp::array<int,1> a({3},1);
    npp::array<int,1> b({3},1);
    npp::array<int,1> c({3},2);

    assert(a==b);
    assert(a!=c);
    std::cout << "[OK] equality\n";
}


void testInitializerConstructor(){

    npp::array<int,2> a{
        {
            {1,2,3},
            {4,5,6}
        }
    };


    assert(a.rank() == 2);
    assert(a.size() == 6);

    auto s = a.shape();

    assert(s[0] == 2);
    assert(s[1] == 3);


    assert(a(0,0) == 1);
    assert(a(0,1) == 2);
    assert(a(0,2) == 3);

    assert(a(1,0) == 4);
    assert(a(1,1) == 5);
    assert(a(1,2) == 6);


    std::cout << "[OK] initializer constructor\n";
}


void testInitializerConstructorFail(){

    bool thrown = false;

    try{
        npp::array<int,2> a{
            {
                {1,2,3},
                {4,5}
            }
        };
    }
    catch(const std::invalid_argument&){
        thrown = true;
    }
    assert(thrown);

    std::cout << "[OK] initializer constructor fail\n";
}



int main(){
    testConstruction();
    testValueConstructor();
    testDouble();
    testNegativeIndexing();
    testNegativeIndexOutOfBounds();
    testStride();
    testStrideAfterReshape();
    testZeroDimension2d();
    testFill();
    testIteratorWrite();
    testConstIterator();
    testStdAlgorithms();
    testData();
    testIndexing();
    testBounds();
    testCopy();
    testEmptyShapes();
    testReshapeBasic();
    testReshapeIndexing();
    testReshape3d();
    testReshapeInvalid();
    testCopyConstructor();
    testCopyAssignment();
    testMoveConstructor();
    testMoveAssignment();
    testSwap();
    testArrayArithmetic();
    testCompoundArithmetic();
    testScalarArithmetic();
    testReverseScalarArithmetic();
    testArithmeticShapeMismatch();
    testSizeOverflow();
    testEquality();
    testInitializerConstructor();
    testInitializerConstructorFail();

    std::cout << "\nALL TESTS PASSED\n";
}