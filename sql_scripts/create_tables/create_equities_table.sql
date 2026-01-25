CREATE TABLE equities (
equity_id bigint unsigned NOT NULL auto_increment,
ticker varchar(10) NOT NULL,
primary key (equity_id),
UNIQUE KEY uq_equities_ticker (ticker)
);



