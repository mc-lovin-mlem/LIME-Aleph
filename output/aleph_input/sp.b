:- use_module(library(lists)).
:- modeh(1, true_class(+example)).
larger(X, Y) :- X > Y.
:- modeb(*, contains(-superpixel, +example)).
%:- modeb(*, contains(#superpixel, +example)).
:- modeb(*, has_color(+superpixel, #color)).
:- modeb(*, has_size(+superpixel, -size)).
:- modeb(*, larger(+size, +size)).
%:- modeb(*, has_name(+superpixel, #name)).
:- modeb(*, on_in_ex(+superpixel, +superpixel, +example)).
:- modeb(*, on_in_ex(+superpixel, +superpixel, +example)).
:- modeb(*, under_in_ex(+superpixel, +superpixel, +example)).
:- modeb(*, under_in_ex(+superpixel, +superpixel, +example)).
:- modeb(*, left_of_in_ex(+superpixel, +superpixel, +example)).
:- modeb(*, left_of_in_ex(+superpixel, +superpixel, +example)).
:- modeb(*, right_of_in_ex(+superpixel, +superpixel, +example)).
:- modeb(*, right_of_in_ex(+superpixel, +superpixel, +example)).
:- modeb(*, top_of_in_ex(+superpixel, +superpixel, +example)).
:- modeb(*, top_of_in_ex(+superpixel, +superpixel, +example)).
:- modeb(*, bottom_of_in_ex(+superpixel, +superpixel, +example)).
:- modeb(*, bottom_of_in_ex(+superpixel, +superpixel, +example)).
:- determination(true_class/1, contains/2).
:- determination(true_class/1, has_color/2).
:- determination(true_class/1, has_size/2).
:- determination(true_class/1, larger/2).
:- determination(true_class/1, on_in_ex/3).
:- determination(true_class/1, on_in_ex/3).
:- determination(true_class/1, under_in_ex/3).
:- determination(true_class/1, under_in_ex/3).
:- determination(true_class/1, left_of_in_ex/3).
:- determination(true_class/1, left_of_in_ex/3).
:- determination(true_class/1, right_of_in_ex/3).
:- determination(true_class/1, right_of_in_ex/3).
:- determination(true_class/1, top_of_in_ex/3).
:- determination(true_class/1, top_of_in_ex/3).
:- determination(true_class/1, bottom_of_in_ex/3).
:- determination(true_class/1, bottom_of_in_ex/3).
:- set(i, 4).
:- set(clauselength, 20).
:- set(minpos, 2).
:- set(minscore, 0).
:- set(verbosity, 0).
:- set(noise, 10).
:- set(nodes, 10000).
:- consult('sp.bk').
