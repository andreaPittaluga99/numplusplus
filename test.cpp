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
        a.at(-3,0);
    }
    catch(const std::out_of_range&){
        thrown = true;
    }

    assert(thrown);

    thrown = false;

    try{
        a.at(0,-4);
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

void testIndexingAt(){
    npp::array<int, 3> a({2,3,4});
    a.at(1,2,3) = 999;
    assert(a.at(1,2,3) == 999);

    std::cout << "[OK] indexing at()\n";
}


void testBounds(){
    npp::array<int, 2> a({3,4});
    bool thrown = false;

    try{
        a.at(3,0);
    }
    catch(const std::out_of_range&){
        thrown = true;
    }

    assert(thrown);

    thrown = false;
    try{
        a.at(0,4);
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

void testBracketOperator(){
    npp::array<int,1> a({5});

    for(std::size_t i = 0; i < a.size(); ++i){
        a[i] = static_cast<int>(i * 10);
    }

    assert(a[0] == 0);
    assert(a[1] == 10);
    assert(a[2] == 20);
    assert(a[3] == 30);
    assert(a[4] == 40);

    std::cout << "[OK] operator[]\n";
}


void testConstBracketOperator(){
    npp::array<int,1> a({3});

    a[0] = 7;
    a[1] = 8;
    a[2] = 9;

    const auto& ca = a;

    assert(ca[0] == 7);
    assert(ca[1] == 8);
    assert(ca[2] == 9);

    std::cout << "[OK] const operator[]\n";
}


void testNegativeBracketOperator(){
    npp::array<int,1> a({5});

    for(std::size_t i = 0; i < a.size(); ++i){
        a[i] = static_cast<int>(i);
    }

    assert(a[-1] == 4);
    assert(a[-2] == 3);
    assert(a[-5] == 0);

    std::cout << "[OK] negative operator[]\n";
}


void testReshaped(){
    npp::array<int,2> a({2,3});

    int value = 0;
    for(auto& x : a){
        x = value++;
    }

    auto b = a.reshaped({3,2});

    assert(b.shape()[0] == 3);
    assert(b.shape()[1] == 2);

    assert(b(0,0) == 0);
    assert(b(0,1) == 1);
    assert(b(1,0) == 2);
    assert(b(1,1) == 3);
    assert(b(2,0) == 4);
    assert(b(2,1) == 5);

    std::cout << "[OK] reshaped\n";
}


void testReshapedDoesNotModifyOriginal(){
    npp::array<int,2> a({2,3});

    int value = 0;
    for(auto& x : a){
        x = value++;
    }

    auto b = a.reshaped({3,2});

    (void)b;

    assert(a.shape()[0] == 2);
    assert(a.shape()[1] == 3);

    assert(a(0,0) == 0);
    assert(a(0,1) == 1);
    assert(a(0,2) == 2);

    assert(a(1,0) == 3);
    assert(a(1,1) == 4);
    assert(a(1,2) == 5);

    std::cout << "[OK] reshaped leaves original unchanged\n";
}


void testReshapedInvalid(){
    npp::array<int,2> a({2,3});

    bool thrown = false;

    try{
        auto b = a.reshaped({4,4});
        (void)b;
    }
    catch(const std::invalid_argument&){
        thrown = true;
    }

    assert(thrown);

    std::cout << "[OK] reshaped invalid\n";
}

void testSum(){
    npp::array<int,2> a({2,3});

    int value = 1;
    for(auto& x : a){
        x = value++;
    }

    assert(a.sum() == 21);

    std::cout << "[OK] sum\n";
}


void testSumEmpty(){
    npp::array<int,2> a({0,5});

    assert(a.sum() == 0);

    std::cout << "[OK] sum empty\n";
}


void testSumNegativeValues(){
    npp::array<int,1> a({5});

    a[0] = -5;
    a[1] = 10;
    a[2] = -3;
    a[3] = 2;
    a[4] = 1;

    assert(a.sum() == 5);

    std::cout << "[OK] sum negative values\n";
}


void testSumSingleElement(){
    npp::array<int,1> a({1},42);

    assert(a.sum() == 42);

    std::cout << "[OK] sum single element\n";
}


void testSumDouble(){
    npp::array<double,2> a({2,2});

    a(0,0) = 0.5;
    a(0,1) = 1.5;
    a(1,0) = 2.0;
    a(1,1) = 4.0;

    assert(a.sum() == 8.0);

    std::cout << "[OK] sum double\n";
}


void testSumAfterReshape(){
    npp::array<int,2> a({2,3});

    int value = 1;
    for(auto& x : a){
        x = value++;
    }

    auto before = a.sum();

    a.reshape({3,2});

    assert(a.sum() == before);

    std::cout << "[OK] sum after reshape\n";
}

void testProd(){
    npp::array<int,2> a({2,3});

    int value = 1;
    for(auto& x : a){
        x = value++;
    }

    assert(a.prod() == 720);

    std::cout << "[OK] prod\n";
}


void testProdEmpty(){
    npp::array<int,2> a({0,5});

    assert(a.prod() == 1);

    std::cout << "[OK] prod empty\n";
}


void testProdNegativeValues(){
    npp::array<int,1> a({5});

    a[0] = -2;
    a[1] = 3;
    a[2] = -1;
    a[3] = 4;
    a[4] = 5;

    assert(a.prod() == 120);

    std::cout << "[OK] prod negative values\n";
}


void testProdSingleElement(){
    npp::array<int,1> a({1}, 42);

    assert(a.prod() == 42);

    std::cout << "[OK] prod single element\n";
}


void testProdDouble(){
    npp::array<double,2> a({2,2});

    a(0,0) = 0.5;
    a(0,1) = 2.0;
    a(1,0) = 4.0;
    a(1,1) = 5.0;

    assert(a.prod() == 20.0);

    std::cout << "[OK] prod double\n";
}


void testProdAfterReshape(){
    npp::array<int,2> a({2,3});

    int value = 1;
    for(auto& x : a){
        x = value++;
    }

    auto before = a.prod();

    a.reshape({3,2});

    assert(a.prod() == before);

    std::cout << "[OK] prod after reshape\n";
}



void testMin(){
    npp::array<int,2> a({2,3});

    int values[] = {5,2,8,1,9,3};

    std::copy(std::begin(values), std::end(values), a.begin());

    assert(a.min() == 1);

    std::cout << "[OK] min\n";
}


void testMax(){
    npp::array<int,2> a({2,3});

    int values[] = {5,2,8,1,9,3};

    std::copy(std::begin(values), std::end(values), a.begin());

    assert(a.max() == 9);

    std::cout << "[OK] max\n";
}


void testMinNegativeValues(){
    npp::array<int,1> a({5});

    a[0] = -5;
    a[1] = 10;
    a[2] = -3;
    a[3] = 2;
    a[4] = 1;

    assert(a.min() == -5);

    std::cout << "[OK] min negative values\n";
}


void testMaxNegativeValues(){
    npp::array<int,1> a({5});

    a[0] = -5;
    a[1] = -10;
    a[2] = -3;
    a[3] = -2;
    a[4] = -1;

    assert(a.max() == -1);

    std::cout << "[OK] max negative values\n";
}


void testMinMaxSingleElement(){
    npp::array<int,1> a({1},42);

    assert(a.min() == 42);
    assert(a.max() == 42);

    std::cout << "[OK] min max single element\n";
}


void testMinMaxDouble(){
    npp::array<double,2> a({2,2});

    a(0,0) = 3.5;
    a(0,1) = -1.2;
    a(1,0) = 8.0;
    a(1,1) = 4.7;

    assert(a.min() == -1.2);
    assert(a.max() == 8.0);

    std::cout << "[OK] min max double\n";
}


void testMinMaxAfterReshape(){
    npp::array<int,2> a({2,3});

    int values[] = {10,4,7,2,9,1};

    std::copy(std::begin(values), std::end(values), a.begin());

    auto min_before = a.min();
    auto max_before = a.max();

    a.reshape({3,2});

    assert(a.min() == min_before);
    assert(a.max() == max_before);

    std::cout << "[OK] min max after reshape\n";
}

void testMean(){
    npp::array<int,2> a({2,3});

    int value = 1;
    for(auto& x : a){
        x = value++;
    }

    assert(a.mean() == 3.5);

    std::cout << "[OK] mean\n";
}


void testMeanEmpty(){
    npp::array<int,2> a({0,5});

    bool thrown = false;

    try{
        auto x = a.mean();
        (void)x;
    }
    catch(const std::invalid_argument&){
        thrown = true;
    }

    assert(thrown);

    std::cout << "[OK] mean empty\n";
}


void testMeanNegativeValues(){
    npp::array<int,1> a({5});

    a[0] = -5;
    a[1] = 10;
    a[2] = -3;
    a[3] = 2;
    a[4] = 1;

    assert(a.mean() == 1.0);

    std::cout << "[OK] mean negative values\n";
}


void testMeanSingleElement(){
    npp::array<int,1> a({1},42);

    assert(a.mean() == 42.0);

    std::cout << "[OK] mean single element\n";
}


void testMeanDouble(){
    npp::array<double,2> a({2,2});

    a(0,0) = 0.5;
    a(0,1) = 1.5;
    a(1,0) = 2.0;
    a(1,1) = 4.0;

    assert(a.mean() == 2.0);

    std::cout << "[OK] mean double\n";
}


void testMeanAfterReshape(){
    npp::array<int,2> a({2,3});

    int value = 1;
    for(auto& x : a){
        x = value++;
    }

    auto before = a.mean();

    a.reshape({3,2});

    assert(a.mean() == before);

    std::cout << "[OK] mean after reshape\n";
}



void testArgminFlattened(){
    npp::array<int,2> a({2,3});

    int values[] = {10, 2, 8, 4, 1, 6};

    std::copy(std::begin(values), std::end(values), a.begin());

    assert(a.argmin_flattened() == 4);

    std::cout << "[OK] argmin flattened\n";
}


void testArgmaxFlattened(){
    npp::array<int,2> a({2,3});

    int values[] = {10, 2, 8, 4, 1, 6};

    std::copy(std::begin(values), std::end(values), a.begin());

    assert(a.argmax_flattened() == 0);

    std::cout << "[OK] argmax flattened\n";
}


void testArgminArgmax1d(){
    npp::array<int,1> a({5});

    a[0] = -5;
    a[1] = 10;
    a[2] = -3;
    a[3] = 2;
    a[4] = 1;

    assert(a.argmin_flattened() == 0);
    assert(a.argmax_flattened() == 1);

    std::cout << "[OK] argmin argmax 1d\n";
}


void testArgminArgmaxNegativeValues(){
    npp::array<int,1> a({5});

    a[0] = -10;
    a[1] = -3;
    a[2] = -20;
    a[3] = -1;
    a[4] = -5;

    assert(a.argmin_flattened() == 2);
    assert(a.argmax_flattened() == 3);

    std::cout << "[OK] argmin argmax negative values\n";
}


void testArgminSingleElement(){
    npp::array<int,1> a({1},42);

    assert(a.argmin_flattened() == 0);
    assert(a.argmax_flattened() == 0);

    std::cout << "[OK] argmin argmax single element\n";
}


void testArgminEmpty(){
    npp::array<int,2> a({0,5});

    bool thrown = false;

    try{
        a.argmin_flattened();
    }
    catch(const std::invalid_argument&){
        thrown = true;
    }

    assert(thrown);

    std::cout << "[OK] argmin empty\n";
}


void testArgmaxEmpty(){
    npp::array<int,2> a({0,5});

    bool thrown = false;

    try{
        a.argmax_flattened();
    }
    catch(const std::invalid_argument&){
        thrown = true;
    }

    assert(thrown);

    std::cout << "[OK] argmax empty\n";
}


void testUnravelIndex2d(){
    npp::array<int,2> a({2,3});

    auto c0 = a.unravel_index(0);

    assert(c0[0] == 0);
    assert(c0[1] == 0);


    auto c4 = a.unravel_index(4);

    assert(c4[0] == 1);
    assert(c4[1] == 1);

    auto c5 = a.unravel_index(5);
    assert(c5[0] == 1);
    assert(c5[1] == 2);

    std::cout << "[OK] unravel index 2d\n";
}


void testUnravelIndex3d(){
    npp::array<int,3> a({2,3,4});

    auto c = a.unravel_index(23);
    assert(c[0] == 1);
    assert(c[1] == 2);
    assert(c[2] == 3);

    std::cout << "[OK] unravel index 3d\n";
}


void testVectorConstructor(){
    std::vector<int> v = {1,2,3,4,5,6};

    npp::array<int,2> a({2,3}, v);

    assert(a.shape()[0] == 2);
    assert(a.shape()[1] == 3);
    assert(a.size() == 6);

    for(std::size_t i = 0; i < v.size(); ++i){
        assert(a.data()[i] == v[i]);
    }

    assert(v.size() == 6);

    std::cout << "[OK] vector constructor\n";
}

void testVectorConstructorMove(){
    std::vector<int> v = {10,20,30,40};

    npp::array<int,2> a({2,2}, std::move(v));

    assert(a.size() == 4);

    assert(a(0,0) == 10);
    assert(a(0,1) == 20);
    assert(a(1,0) == 30);
    assert(a(1,1) == 40);

    std::cout << "[OK] vector move constructor\n";
}

void testVectorConstructorWrongSize(){
    bool thrown = false;

    try{
        std::vector<int> v = {1,2,3};

        npp::array<int,2> a({2,2}, v);
    }
    catch(const std::invalid_argument&){
        thrown = true;
    }

    assert(thrown);

    std::cout << "[OK] vector constructor wrong size\n";
}

void testVectorConstructorEmpty(){
    std::vector<int> v;

    npp::array<int,2> a({0,5}, v);

    assert(a.empty());
    assert(a.size() == 0);
    assert(a.begin() == a.end());

    std::cout << "[OK] vector constructor empty\n";
}

void testArrayConstructor(){
    std::array<int,6> data = {1,2,3,4,5,6};

    npp::array<int,2> a({2,3}, data);

    assert(a.shape()[0] == 2);
    assert(a.shape()[1] == 3);
    assert(a.size() == 6);

    for(std::size_t i = 0; i < data.size(); ++i){
        assert(a.data()[i] == data[i]);
    }

    std::cout << "[OK] array constructor\n";
}

void testArrayConstructorWrongSize(){
    bool thrown = false;

    try{
        std::array<int,5> data = {1,2,3,4,5};

        npp::array<int,2> a({2,3}, data);
    }
    catch(const std::invalid_argument&){
        thrown = true;
    }

    assert(thrown);

    std::cout << "[OK] array constructor wrong size\n";
}

void testArrayConstructorEmpty(){
    std::array<int,0> data{};

    npp::array<int,2> a({0,5}, data);

    assert(a.empty());
    assert(a.size() == 0);
    assert(a.begin() == a.end());

    std::cout << "[OK] array constructor empty\n";
}

void testMathAbs(){
    npp::array<int,1> a({5});

    a[0] = -5;
    a[1] = 2;
    a[2] = -3;
    a[3] = 0;
    a[4] = 10;

    auto b = a.abs();

    assert(b[0] == 5);
    assert(b[1] == 2);
    assert(b[2] == 3);
    assert(b[3] == 0);
    assert(b[4] == 10);
    assert(a[0] == -5);

    std::cout << "[OK] abs\n";
}

void testMathSqrt(){
    npp::array<double,1> a({4});

    a[0] = 0.0;
    a[1] = 1.0;
    a[2] = 4.0;
    a[3] = 9.0;

    auto b = a.sqrt();

    assert(b[0] == 0.0);
    assert(b[1] == 1.0);
    assert(b[2] == 2.0);
    assert(b[3] == 3.0);

    std::cout << "[OK] sqrt\n";
}

void testMathExp(){
    npp::array<double,1> a({4});

    a[0] = 0.0;
    a[1] = 1.0;
    a[2] = 2.0;
    a[3] = -1.0;

    auto b = a.exp();

    assert(std::abs(b[0] - 1.0) < 1e-12);
    assert(std::abs(b[1] - std::exp(1.0)) < 1e-12);
    assert(std::abs(b[2] - std::exp(2.0)) < 1e-12);
    assert(std::abs(b[3] - std::exp(-1.0)) < 1e-12);

    assert(a[2] == 2.0);

    std::cout << "[OK] exp\n";
}

void testMathLog(){
    npp::array<double,1> a({4});

    a[0] = 1.0;
    a[1] = 2.0;
    a[2] = 10.0;
    a[3] = std::exp(1.0);

    auto b = a.log();

    assert(std::abs(b[0] - 0.0) < 1e-12);
    assert(std::abs(b[1] - std::log(2.0)) < 1e-12);
    assert(std::abs(b[2] - std::log(10.0)) < 1e-12);
    assert(std::abs(b[3] - 1.0) < 1e-12);
    assert(a[3] == std::exp(1.0));

    std::cout << "[OK] log\n";
}

void testMathLog10(){
    npp::array<double,1> a({4});
    a[0] = 1.0;
    a[1] = 10.0;
    a[2] = 100.0;
    a[3] = 1000.0;
    auto b = a.log10();
    assert(std::abs(b[0] - 0.0) < 1e-12);
    assert(std::abs(b[1] - 1.0) < 1e-12);
    assert(std::abs(b[2] - 2.0) < 1e-12);
    assert(std::abs(b[3] - 3.0) < 1e-12);

    std::cout << "[OK] log10\n";
}
void testMathSin(){
    npp::array<double,1> a({4});
    constexpr double pi = 3.14159265358979323846;
    a[0] = 0.0;
    a[1] = pi / 2.0;
    a[2] = pi;
    a[3] = -pi / 2.0;

    auto b = a.sin();
    assert(std::abs(b[0] - 0.0) < 1e-12);
    assert(std::abs(b[1] - 1.0) < 1e-12);
    assert(std::abs(b[2] - 0.0) < 1e-12);
    assert(std::abs(b[3] + 1.0) < 1e-12);

    std::cout << "[OK] sin\n";
}
void testMathCos(){
    npp::array<double,1> a({4});
    constexpr double pi = 3.14159265358979323846;
    a[0] = 0.0;
    a[1] = pi / 2.0;
    a[2] = pi;
    a[3] = -pi / 2.0;

    auto b = a.cos();
    assert(std::abs(b[0] - 1.0) < 1e-12);
    assert(std::abs(b[1] - 0.0) < 1e-12);
    assert(std::abs(b[2] + 1.0) < 1e-12);
    assert(std::abs(b[3] - 0.0) < 1e-12);

    std::cout << "[OK] cos\n";
}

void testMathTan(){
    npp::array<double,1> a({4});
    constexpr double pi = 3.14159265358979323846;

    a[0] = 0.0;
    a[1] = pi / 4.0;
    a[2] = -pi / 4.0;
    a[3] = pi;

    auto b = a.tan();
    assert(std::abs(b[0] - 0.0) < 1e-12);
    assert(std::abs(b[1] - 1.0) < 1e-12);
    assert(std::abs(b[2] + 1.0) < 1e-12);
    assert(std::abs(b[3] - 0.0) < 1e-12);

    std::cout << "[OK] tan\n";
}

void testBoolArray(){

    npp::array<bool,1> a({4});

    a[0]=true;
    a[1]=false;
    a[2]=true;
    a[3]=false;

    assert(a[0]);
    assert(!a[1]);
    assert(a[2]);
    assert(!a[3]);

    std::cout << "[OK] bool array\n";
}

void testScalarGreater(){
    npp::array<int,1> a({5});

    a[0] = 1;
    a[1] = 5;
    a[2] = 6;
    a[3] = 3;
    a[4] = 10;

    auto b = a > 5;

    static_assert(std::is_same_v<decltype(b), npp::array<bool,1>>);

    assert(b[0] == false);
    assert(b[1] == false);
    assert(b[2] == true);
    assert(b[3] == false);
    assert(b[4] == true);

    std::cout << "[OK] scalar >\n";
}

void testScalarLess(){
    npp::array<int,1> a({5});

    a[0] = 1;
    a[1] = 5;
    a[2] = 6;
    a[3] = 3;
    a[4] = 10;

    auto b = a < 5;

    static_assert(std::is_same_v<decltype(b), npp::array<bool,1>>);

    assert(b[0] == true);
    assert(b[1] == false);
    assert(b[2] == false);
    assert(b[3] == true);
    assert(b[4] == false);

    std::cout << "[OK] scalar <\n";
}

void testScalarGreaterEqual(){
    npp::array<int,1> a({5});

    a[0] = 1;
    a[1] = 5;
    a[2] = 6;
    a[3] = 5;
    a[4] = 10;

    auto b = a >= 5;

    static_assert(std::is_same_v<decltype(b), npp::array<bool,1>>);

    assert(b[0] == false);
    assert(b[1] == true);
    assert(b[2] == true);
    assert(b[3] == true);
    assert(b[4] == true);

    std::cout << "[OK] scalar >=\n";
}

void testScalarLessEqual(){
    npp::array<int,1> a({5});

    a[0] = 1;
    a[1] = 5;
    a[2] = 6;
    a[3] = 3;
    a[4] = 10;

    auto b = a <= 5;

    static_assert(std::is_same_v<decltype(b), npp::array<bool,1>>);

    assert(b[0] == true);
    assert(b[1] == true);
    assert(b[2] == false);
    assert(b[3] == true);
    assert(b[4] == false);

    std::cout << "[OK] scalar <=\n";
}

void testScalarEqual(){
    npp::array<int,1> a({5});

    a[0] = 1;
    a[1] = 5;
    a[2] = 6;
    a[3] = 5;
    a[4] = 10;

    auto b = a == 5;

    static_assert(std::is_same_v<decltype(b), npp::array<bool,1>>);

    assert(b[0] == false);
    assert(b[1] == true);
    assert(b[2] == false);
    assert(b[3] == true);
    assert(b[4] == false);

    std::cout << "[OK] scalar ==\n";
}

void testScalarNotEqual(){
    npp::array<int,1> a({5});

    a[0] = 1;
    a[1] = 5;
    a[2] = 6;
    a[3] = 5;
    a[4] = 10;

    auto b = a != 5;

    static_assert(std::is_same_v<decltype(b), npp::array<bool,1>>);

    assert(b[0] == true);
    assert(b[1] == false);
    assert(b[2] == true);
    assert(b[3] == false);
    assert(b[4] == true);

    std::cout << "[OK] scalar !=\n";
}


void testZeros(){
    auto a = npp::array<int,2>::zeros({2,3});

    assert((a.shape() == std::array<std::size_t,2>{2,3}));
    assert(a.size() == 6);

    for(auto x : a){
        assert(x == 0);
    }

    std::cout << "[OK] zeros\n";
}


void testOnes(){
    auto a = npp::array<int,2>::ones({2,3});

    assert((a.shape() == std::array<std::size_t,2>{2,3}));
    assert(a.size() == 6);

    for(auto x : a){
        assert(x == 1);
    }

    std::cout << "[OK] ones\n";
}


void testFull(){
    auto a = npp::array<int,3>::full({2,2,2}, 7);

    assert((a.shape() == std::array<std::size_t,3>{2,2,2}));
    assert(a.size() == 8);

    for(auto x : a){
        assert(x == 7);
    }

    std::cout << "[OK] full\n";
}


void testIdentity(){
    auto a = npp::array<double,2>::identity(3);

    static_assert(std::is_same_v<decltype(a), npp::array<double,2>>);

    assert((a.shape() == std::array<std::size_t,2>{3,3}));

    assert(a(0,0) == 1.0);
    assert(a(0,1) == 0.0);
    assert(a(0,2) == 0.0);

    assert(a(1,0) == 0.0);
    assert(a(1,1) == 1.0);
    assert(a(1,2) == 0.0);

    assert(a(2,0) == 0.0);
    assert(a(2,1) == 0.0);
    assert(a(2,2) == 1.0);

    std::cout << "[OK] identity\n";
}


void testLinspace(){
    auto a = npp::array<double,1>::linspace(0.0, 1.0, 5);

    static_assert(std::is_same_v<decltype(a), npp::array<double,1>>);

    assert(a.size() == 5);

    assert(a[0] == 0.0);
    assert(a[1] == 0.25);
    assert(a[2] == 0.5);
    assert(a[3] == 0.75);
    assert(a[4] == 1.0);

    std::cout << "[OK] linspace\n";
}


void testLinspaceSingle(){
    auto a = npp::array<float,1>::linspace(3.0f, 10.0f, 1);

    assert(a.size() == 1);
    assert(a[0] == 3.0f);

    std::cout << "[OK] linspace single\n";
}


void testLinspaceEmpty(){
    auto a = npp::array<double,1>::linspace(0.0, 1.0, 0);

    assert(a.empty());
    assert(a.size() == 0);

    std::cout << "[OK] linspace empty\n";
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
    testBracketOperator();
    testConstBracketOperator();
    testNegativeBracketOperator();
    testReshaped();
    testReshapedDoesNotModifyOriginal();
    testReshapedInvalid();
    testSum();
    testSumEmpty();
    testSumNegativeValues();
    testSumSingleElement();
    testSumDouble();
    testSumAfterReshape();
    testProd();
    testProdEmpty();
    testProdNegativeValues();
    testProdSingleElement();
    testProdDouble();
    testProdAfterReshape();
    testMin();
    testMax();
    testMinNegativeValues();
    testMaxNegativeValues();
    testMinMaxSingleElement();
    testMinMaxDouble();
    testMinMaxAfterReshape();
    testMean();
    testMeanEmpty();
    testMeanNegativeValues();
    testMeanSingleElement();
    testMeanDouble();
    testMeanAfterReshape();
    testArgminFlattened();
    testArgmaxFlattened();
    testArgminArgmax1d();
    testArgminArgmaxNegativeValues();
    testArgminSingleElement();
    testArgminEmpty();
    testArgmaxEmpty();
    testUnravelIndex2d();
    testUnravelIndex3d();
    testVectorConstructor();
    testVectorConstructorMove();
    testVectorConstructorWrongSize();
    testVectorConstructorEmpty();
    testArrayConstructor();
    testArrayConstructorWrongSize();
    testArrayConstructorEmpty();
    testMathAbs();
    testMathSqrt();
    testMathExp();
    testMathLog();
    testMathLog10();
    testMathSin();
    testMathCos();
    testMathTan();
    testBoolArray();
    testScalarGreater();
    testScalarLess();
    testScalarGreaterEqual();
    testScalarLessEqual();
    testScalarEqual();
    testScalarNotEqual();
    testZeros();
    testOnes();
    testFull();
    testIdentity();
    testLinspace();
    testLinspaceSingle();
    testLinspaceEmpty();


    std::cout << "\nALL TESTS PASSED\n";
}