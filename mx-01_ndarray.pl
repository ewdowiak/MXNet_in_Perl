#!/usr/bin/env perl 

##  Copyright 2019 Eryk Wdowiak
##  
##  Licensed under the Apache License, Version 2.0 (the "License");
##  you may not use this file except in compliance with the License.
##  You may obtain a copy of the License at
##  
##      http://www.apache.org/licenses/LICENSE-2.0
##  
##  Unless required by applicable law or agreed to in writing, software
##  distributed under the License is distributed on an "AS IS" BASIS,
##  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
##  See the License for the specific language governing permissions and
##  limitations under the License.

##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  

##  Perl replication of:
##  https://beta.mxnet.io/guide/crash-course/1-ndarray.html

##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  

use strict;
use warnings;
use AI::MXNet qw(mx);

##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##

print "\n";
print "Manipulate data with NDArray\n\n";
print "========================================\n\n";
print "Get started\n";
print "\n";

my $ndarray = nd->array([[1,2,3],[5,6,7]]);
print 'my $ndarray = nd->array([[1,2,3],[5,6,7]]);';
print $ndarray->aspdl;
print "\n";

my $xx = nd->ones([2,3]);
print 'my $xx = nd->ones([2,3]);';
print $xx->aspdl;
print "\n";

my $yy = nd->random->uniform(-1,1,[2,3]);
print 'my $yy = nd->random->uniform(-1,1,[2,3]);';
print $yy->aspdl;
print "\n";

my $zz = nd->full([2,3],2);
print 'my $zz = nd->full([2,3],2);';
print $zz->aspdl;
print "shape: ". '['. join(",",@{$ndarray->shape}) .']'."\n";
print "size:  ". $ndarray->size ."\n";
print "dtype: ". $ndarray->dtype ."\n";
print "\n";

##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##

print "========================================\n\n";
print "Operations\n";
print "\n";

my $zy = $zz * $yy;
print 'my $zy = $zz * $yy;';
print $zy->aspdl;
print "\n";

my $ye = $yy->exp;
print 'my $ye = $yy->exp;';
print $ye->aspdl;
print "\n";

my $zyt = nd->dot($zz,$yy->T);
print 'my $zyt = nd->dot($zz,$yy->T);';
print $zyt->aspdl;
print "\n";

##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##

print "========================================\n\n";
print "Indexing\n";
print "\n\n";

print 'here is $yy again';
print $yy->aspdl;
print "\n";
print "get element in last row, last column\n";
print '$yy->slice(1,2)';
print $yy->slice(1,2)->aspdl;
print "\n\n";
print "change that element\n";
print '$yy->slice(1,2) .= 7;';
$yy->slice(1,2) .= 7;
print $yy->aspdl;
print "\n\n";
print "to change every value in a row\n";
print '$yy->slice(0,) .= 14;';
$yy->slice(0,) .= 14;
print $yy->aspdl;
print "\n\n";
print "BUG:  unable to change every value in a column\n";
print "and unable to select a _range_ of rows or columns\n";
print "\n";

##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##

print "========================================\n\n";
print "Converting between MXNet NDArray and PDL\n";
print "\n";
print 'when working with the array as a "pdl," '."\n";
print 'remember that PDL is  COLUMN-MAJOR'."\n";
print "\n";
print "get second and third columns\n";
print '$yy->aspdl->slice(\'1:2,:\')'."\n";
print $yy->aspdl->slice('1:2,:');
print "\n";
print "change the value of the second and third columns\n";
print "(by converting to PDL and then back to NDArray)\n";
{   my $tmp = $yy->aspdl;
    $tmp->slice('1:2,:') .= 21;
    $yy = nd->array($tmp);
}
print $yy->aspdl;
print "\n";
print "change the values in the first two columns of the second row\n";
print "(by converting to PDL and then back to NDArray)\n";
{   my $tmp = $yy->aspdl;
    $tmp->slice('0:1,1') .= 42;
    $yy = nd->array($tmp);
}
print $yy->aspdl;
print "\n";
print "\n";


##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##
  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##
##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  

##  SUBROUTINES

